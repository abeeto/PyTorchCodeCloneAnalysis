import argparse
from torch import profiler
from torch._C._autograd import DeviceType
# from torch._C._profiler import _ProfilerEvent, _EventType
# from torch.profiler._pattern_matcher import eventTreeBFS
from common_func import *
from profile_event import ProfileEventSlim, TraceEvent
import torch
import json
from analysis_result import AnalysisResult
import numpy as np
from occupancy_calculator import CudaOccupancyCalculator


KINETO_EVENT_ON_CPU = [
    "cudaDeviceGetStreamPriorityRange",
    "cudaStreamGetPriority",
    # "cudaDeviceSynchronize",
    "cudaStreamIsCapturing",
    "cudaFuncGetAttributes",
    "cudaStreamWaitEvent",
    "cudaLaunchKernel",
    "cudaFuncSetAttribute",
]


class TorchExpert:
    """
    This class is used to profile the model and do the analysis.
    Attribute:
        prof: the profiler reference
        events_raw: the raw events from the profiler.
        profiler_config: the config for profiling
        json_trace: the json file of the profiling result
        cuda_kernels: cuda kernels, ProfilerEventSlim
    TorchExpert requires PyTorch >= August 8st, 2022
    """

    def __init__(self, model_name='', output_csv_file=None, analyze_json_only=False, profiler_folder='./logs/'):
        self.prof = None
        self.events_raw = []
        self.event_tree_roots = []
        self.events_kineto = []
        self.profiler_config = {
            "activities": [profiler.ProfilerActivity.CUDA, profiler.ProfilerActivity.CPU],
            "profile_detailed": True,
            "profile_folder": profiler_folder,
            "nwarmup": 3
        }
        self.json_trace = None
        self.analysis_result = None
        self.analyze_json_only = analyze_json_only
        self.model_name = model_name
        self.output_csv_file = output_csv_file
        self.occup_calc = CudaOccupancyCalculator("8.0")


    def set_profile_config(self, activity_groups, profile_detailed, profile_folder, nwarmup):
        self.profiler_config = {
            "activities": activity_groups,
            "profile_detailed": profile_detailed,
            "profile_folder": profile_folder,
            "nwarmup": nwarmup
        }

    def set_profile(self, prof):
        """
        If the profiling happens outside this class, you can set the profile reference here.
        """
        self.prof = prof
        # _ProfileEvent can't provide enough information, so we need to revert back to kineto events
        # self.event_tree_roots = prof.profiler.kineto_results.experimental_event_tree()
        # self.events_raw = list(eventTreeBFS(self.event_tree_roots))
        self.events_kineto = prof.profiler.kineto_results.events()
        self.events_raw = self.events_kineto

    def profile(self, func, *args, **kwargs):
        """
        This function is used to profile the model. It is not necessary to call this function. 
        You can directly use the profiler outside this class.
        """
        nwarmup = int(self.profiler_config["nwarmup"])
        with profiler.profile(
            schedule=profiler.schedule(wait=0, warmup=nwarmup, active=1),
            activities=self.profiler_config["activities"],
            record_shapes=self.profiler_config["profile_detailed"],
            profile_memory=self.profiler_config["profile_detailed"],
            with_stack=self.profiler_config["profile_detailed"],
            with_flops=self.profiler_config["profile_detailed"],
            on_trace_ready=profiler.tensorboard_trace_handler(
                self.profiler_config["profile_folder"]),
        ) as prof:
            for _i in range(nwarmup + 1):
                func(*args, **kwargs)
                # Need to sync here to match run_one_step()'s timed run.
                torch.cuda.synchronize()
                # The last call of prof.step() will clean the profile,
                # so ignore it in the last iteration.
                if _i != nwarmup:
                    prof.step()
        # print(prof.key_averages(group_by_input_shape=True).table(
        #     sort_by="cpu_time_total", row_limit=30))
        self.set_profile(prof)

    def get_all_idleness(self, events):
        """
        This function is used to get the idleness of the events.
        Args:
            events: a sorted list of events by start time
        Returns:
            a list of idleness event
        """
        if len(events) == 0:
            return []
        idle_events = []
        last_end_time_ns = events[0].end_time_ns
        for i in range(1, len(events)):
            duration = events[i].start_time_ns - last_end_time_ns
            # ignore the idleness less than 0.01ms
            if duration > 0.01 * 1e6:
                idle_events.append(ProfileEventSlim(
                    event=None, duration_time_ns=events[i].start_time_ns - last_end_time_ns, start_time_ns=last_end_time_ns, end_time_ns=events[i].start_time_ns))
        return idle_events

    def get_events_from_json(self):
        slimevents = []
        end_time_ns = 0
        start_time_ns = 0
        memcpy_time = 0
        for event in self.json_trace['traceEvents']:
            if event.get('cat', '').lower() == 'kernel' or event.get('cat', '').lower() == 'gpu_memcpy':
                dur = event['dur']*1e3
                ts = event['ts']*1e3
                te = ts + dur
                slimevents.append(ProfileEventSlim(
                    event=None, duration_time_ns=dur, start_time_ns=ts, end_time_ns=te))
                end_time_ns = max(end_time_ns, te)
                if start_time_ns == 0:
                    start_time_ns = ts
                else:
                    start_time_ns = min(start_time_ns, ts)
                if event.get('cat', '') == 'gpu_memcpy':
                    memcpy_time += dur
        return slimevents, start_time_ns, end_time_ns, memcpy_time

    def get_events_from_profile(self):
        slimevents = []
        end_time_ns = 0
        start_time_ns = self.events_raw[0].start_us() * 1e3 if len(
            self.events_raw) else 0
        memcpy_time = 0
        for event in self.events_raw:
            if event.device_type() == DeviceType.CUDA:
                slimevents.append(ProfileEventSlim(event))
                # @Future: Update to _ProfilerEvent. The kineto event only has us resolution.
                end_time_ns = max(
                    end_time_ns, (event.start_us() + event.duration_us())*1e3)
                start_time_ns = min(start_time_ns, event.start_us() * 1e3)
                if event.name().strip().startswith("Memcpy"):
                    memcpy_time += event.duration_us() * 1e3
        return slimevents, start_time_ns, end_time_ns, memcpy_time

    def analyze(self, json_path='./'):
        """
        This function is used to analyze the profiling result. Will be changed to add more features in the future.
        """
        print("\n\n")
        self.load_json(json_path)
        if self.analyze_json_only:
            slimevents, start_time_ns, end_time_ns, memcpy_time = self.get_events_from_json()
        else:
            slimevents, start_time_ns, end_time_ns, memcpy_time = self.get_events_from_profile()
        merged_slimevents = merge_interval(slimevents)

        # get all idleness
        # @TODO: the results are not correct
        idle_events = self.get_all_idleness(merged_slimevents)
        # get all kernels' occupancy
        avg_kernel_occupancy = self.get_avg_kernel_occupancy()
        sum_gpu_busy = 0
        for slimevent in merged_slimevents:
            # print(slimevent.start_us, slimevent.end_us)
            sum_gpu_busy += slimevent.end_time_ns - slimevent.start_time_ns
            # for event in slimevent.include_events:
            #     print(event.name())
        if start_time_ns == 0:
            print("Error: No events found.")
            return
        app_duration = end_time_ns - start_time_ns
        self.analysis_result = AnalysisResult(
            app_duration=app_duration, memcpy_time=memcpy_time, gpu_busy_time=sum_gpu_busy, avg_kernel_occupancy=avg_kernel_occupancy, model_name=self.model_name, output_csv_file=self.output_csv_file)
        # self.analysis_result.print_as_str()
        self.analysis_result.print_as_csv()

    def load_json(self, json_path):
        """
        This function is used to load the profiling result from a json file.
        Args:
            json_file: the path of the json file
        """
        # check if json_path is a file or a folder
        if os.path.isfile(json_path):
            json_file = json_path
        else:
            json_file = get_latest_file(json_path)
        if json_file is None:
            print("Error: No json file found.")
            return
        print("Analyzing json file: {}".format(json_file))
        with open(json_file, "r") as f:
            self.json_trace = json.load(f)

    def get_avg_kernel_occupancy(self):
        """
        This function is used to get the occupancy of all kernels in the trace file.
        Returns:
            a dictionary of kernel occupancy
        """
        sum_duration = 0
        kernel_occupancies = []
        for event in self.json_trace['traceEvents']:
            if event.get('cat', '').lower() == 'kernel':
                duration = event['dur']*1e3
                block_size = np.prod(event['args']['block'])
                reg_per_thread = event['args']['registers per thread']
                smem = event['args'].get('shared memory', 0)
                self.occup_calc.set_inputs(block_size, reg_per_thread, "8.0", smem)
                occupancy = self.occup_calc.occupancyOfMultiprocessor()
                occupancy_in_trace = event['args'].get('est. achieved occupancy %', 0)
                # if occupancy*100 !gccgjudnvkdcghjcetthjvkeggdnkggicy in the trace file: ", occupancy_in_trace)
                kernel_occupancies.append(occupancy*duration)
                sum_duration += duration
                
        # print("kernel_occupancies: ", kernel_occupancies)
        avg_occupancy = sum(kernel_occupancies)/sum_duration * 100 if sum_duration > 0 else 0
        return avg_occupancy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default='./', help="the path of the json file or the folder containing the json files")
    parser.add_argument("--model_name", type=str, default='model', help="the name of the model")
    parser.add_argument("--output_csv_file", type=str, default='analysis_result.csv', help="the name of the output csv file")
    parser.add_argument("--analyze_json_only", type=bool, default=True, help="If True, will only analyze the json file. If False, will do the profiling and analysis of the json trace file.")
    parser.add_argument("--profiler_folder", type=str, default='./logs/', help="the folder to save the PyTorch profiler results")
    args = parser.parse_args()
    torchexpert = TorchExpert(model_name=args.model_name, output_csv_file=args.output_csv_file, analyze_json_only=args.analyze_json_only, profiler_folder=args.profiler_folder)
    torchexpert.analyze(args.json_path)
    