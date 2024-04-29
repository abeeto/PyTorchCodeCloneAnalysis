class AnalysisResult:
    """
    A class to store the analysis result.
    app_duration: the duration of the whole application
    gpu_busy_time: the duration when the GPU has at least one active kernel or memcpy
    memcpy_time: the duration for memcpy
    gpu_active_time: the duration when the GPU has at least one active kernel
    """

    def __init__(self, memcpy_time=0, app_duration=0, gpu_busy_time=0, avg_kernel_occupancy=0, model_name="", output_csv_file=None):
        self.memcpy_time = memcpy_time
        self.app_duration = app_duration
        self.gpu_active_time = gpu_busy_time - memcpy_time
        self.gpu_busy_time = gpu_busy_time
        self.avg_kernel_occupancy = avg_kernel_occupancy
        self.model_name = model_name
        self.output_csv_file = output_csv_file

    def print_as_str(self):
        print("\nModel: %s" % self.model_name)
        print("{:<25} {:>10}".format(
            "GPU memcpy time", "%.2fms" % (self.memcpy_time / 1e6)))
        print("{:<25} {:>10}".format(
            "GPU active time", "%.2fms" % (self.gpu_active_time / 1e6)))
        print("{:<25} {:>10}".format(
            "GPU busy time", "%.2fms" % (self.gpu_busy_time / 1e6)))
        print("{:<25} {:>10}".format(
            "GPU total time:", "%.2fms" % (self.app_duration / 1e6)))
        print("{:<25} {:>10}".format("GPU memcpy time ratio:", "%.2f%%" %
                                     (self.memcpy_time * 100 / self.app_duration)))
        print("{:<25} {:>10}".format("GPU active time ratio:", "%.2f%%" %
                                     (self.gpu_active_time * 100 / self.app_duration)))
        print("{:<25} {:>10}".format("GPU busy time ratio:", "%.2f%%" %
                                     (self.gpu_busy_time * 100 / self.app_duration)))
        print("{:<25} {:>10}".format("Average kernel occupancy:", "%.2f%%" %
                                     (self.avg_kernel_occupancy)))

    def print_as_csv(self):
        # print("Model, memcpy, active, busy, total, memcpy ratio, active ratio, busy ratio, average occupancy")
        if self.output_csv_file:
            out_str = "%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n" % (
                self.model_name,
                self.memcpy_time / 1e6,
                self.gpu_active_time / 1e6,
                self.gpu_busy_time / 1e6,
                self.app_duration / 1e6,
                self.memcpy_time * 100 / self.app_duration,
                self.gpu_active_time * 100 / self.app_duration,
                self.gpu_busy_time * 100 / self.app_duration,
                self.avg_kernel_occupancy)
            with open(self.output_csv_file, "a") as f:
                f.write(out_str)
        else:
            print("output_csv_file is not set")
