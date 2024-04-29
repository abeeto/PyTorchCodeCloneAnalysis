import torch
import sys
import time
import argparse

from torchvision import datasets, models, transforms
from transformers import AutoTokenizer, AutoModel # BERT
from transformers import GPT2Tokenizer, GPT2Model # GPT2
#from dlrm_s_pytorch import DLRM_Net # DLRM

def average_90_percent(l):
    # average over iterable
    # check only last ~90% of elements (warmup)
    if (len(l) < 10):
        print("average(): I need at least 10 items to work with!")
        return 0
    better_l = l[int(len(l)/10):]
    return sum(better_l) / len(better_l)

def sec_to_ms(sec):
    # a.bcdefghijk... --> "abcd.efg"
    return str(int(sec * 10**6) / 10**3)

def bytes_to_mib(bytes):
    # 50*1024*1024 --> "50.000"
    return str(int((bytes * 10**3) / 2**20) / 10**3)

def main():
    #print("MODEL NAME, BATCH SIZE, AVG LATENCY (ms), AVG MEM USAGE (MiB)")
    #parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--num_inference', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--gpu', action="store_true", default=False)
    args = parser.parse_args()
    model_name = args.model_name
    num_inference = args.num_inference
    batch_size = args.batch_size
    use_gpu = args.gpu and torch.cuda.is_available()
    # stores latency / memory usage values
    l_inference_latency = list()
    l_memory_capacity = list()
    # call corresponding DNN model...
    # TODO: ADD OTHER MODELS - RESNET50, ...
    # TODO: FIX NLP MODELS' SEQUENCE LENGTH
    if (model_name == "resnet18"):
        with torch.no_grad():
            model = models.resnet18(True, True)
            if use_gpu:
                model = model.cuda()
            # inference
            for i in range(num_inference):
                # input
                inputs = torch.zeros(batch_size, 3, 224, 224)
                if use_gpu:
                    inputs = inputs.to('cuda')
                start_time = time.time()
                _ = model(inputs)
                torch.cuda.synchronize()
                end_time = time.time()
                l_inference_latency.append(end_time - start_time)
                l_memory_capacity.append(torch.cuda.memory_allocated())
            str_avg_inf_time = sec_to_ms(average_90_percent(l_inference_latency))
            str_avg_mem_usage = bytes_to_mib(average_90_percent(l_memory_capacity))
            print(",".join(["RESNET18", str(batch_size), str_avg_inf_time, str_avg_mem_usage]))

    elif (model_name == "wide_resnet101_2"):
        with torch.no_grad():
            model = models.wide_resnet101_2(True, True)
            if use_gpu:
                model = model.cuda()
            # inference
            for i in range(num_inference):
                # input
                inputs = torch.zeros(batch_size, 3, 224, 224)
                if use_gpu:
                    inputs = inputs.to('cuda')
                start_time = time.time()
                _ = model(inputs)
                torch.cuda.synchronize()
                end_time = time.time()
                l_inference_latency.append(end_time - start_time)
                l_memory_capacity.append(torch.cuda.memory_allocated())
            str_avg_inf_time = sec_to_ms(average_90_percent(l_inference_latency))
            str_avg_mem_usage = bytes_to_mib(average_90_percent(l_memory_capacity))
            print(",".join(["WIDE-RESNET101-2", str(batch_size), str_avg_inf_time, str_avg_mem_usage]))

    elif (model_name == "mobilenet"):
        with torch.no_grad():
            model = models.mobilenet_v2(True, True)
            if use_gpu:
                model = model.cuda()
            # warmup
            for i in range(num_inference):
                inputs = torch.zeros(batch_size, 3, 224, 224)
                if use_gpu:
                    inputs = inputs.to('cuda')
                start_time = time.time()
                _ = model(inputs)
                torch.cuda.synchronize()
                end_time = time.time()
                l_inference_latency.append(end_time - start_time)
                l_memory_capacity.append(torch.cuda.memory_allocated())
            str_avg_inf_time = sec_to_ms(average_90_percent(l_inference_latency))
            str_avg_mem_usage = bytes_to_mib(average_90_percent(l_memory_capacity))
            print(",".join(["MOBILENET_V2", str(batch_size), str_avg_inf_time, str_avg_mem_usage]))

    elif (model_name == "bert"):
        with torch.no_grad():
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            model = AutoModel.from_pretrained("bert-base-uncased")
            if use_gpu:
                model = model.cuda()
            # inference
            for i in range(num_inference):
                # BERT maximum sequence length 512
                sample_text = "BERT" * int(512/4)
                texts = [sample_text] * batch_size
                inputs = tokenizer(texts, return_tensors="pt")
                if use_gpu:
                    inputs = inputs.to('cuda')
                start_time = time.time()
                _ = model(**inputs)
                torch.cuda.synchronize()
                end_time = time.time()
                l_inference_latency.append(end_time - start_time)
                l_memory_capacity.append(torch.cuda.memory_allocated())
            str_avg_inf_time = sec_to_ms(average_90_percent(l_inference_latency))
            str_avg_mem_usage = bytes_to_mib(average_90_percent(l_memory_capacity))
            print(",".join(["BERT-BASE-UNCASED", str(batch_size), str_avg_inf_time, str_avg_mem_usage]))

    elif (model_name == "gpt2"):
        with torch.no_grad():
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            model = GPT2Model.from_pretrained("gpt2")
            if use_gpu:
                model = model.cuda()
            # inference
            for i in range(num_inference):
                # GPT2 maximum sequence length 124
                sample_text = "GPT2" * int(1024/4)
                texts = [sample_text] * batch_size
                inputs = tokenizer(texts, return_tensors="pt")
                if use_gpu:
                    inputs = inputs.to('cuda')
                start_time = time.time()
                _ = model(**inputs)
                torch.cuda.synchronize()
                end_time = time.time()
                l_inference_latency.append(end_time - start_time)
                l_memory_capacity.append(torch.cuda.memory_allocated())
            str_avg_inf_time = sec_to_ms(average_90_percent(l_inference_latency))
            str_avg_mem_usage = bytes_to_mib(average_90_percent(l_memory_capacity))
            print(",".join(["GPT2", str(batch_size), str_avg_inf_time, str_avg_mem_usage]))
    
    elif (model_name == "dlrm"):
        print("Unimplemented model: DLRM")
        # TODO: MAKE IT WORK... PLEASE
        '''
        with torch.no_grad():
            model = DLRM_Net()
            if use_gpu:
                model = model.cuda()
            # inference
            for i in range(num_inference):
                inputs = ????
                if use_gpu:
                    inputs = inputs.to('cuda')
                start_time = time.time()
                _ = model(**inputs)
                torch.cuda.synchronize()
                end_time = time.time()
                l_inference_latency.append(end_time - start_time)
                l_memory_capacity.append(torch.cuda.memory_allocated())
            str_avg_inf_time = sec_to_ms(average_90_percent(l_inference_latency))
            str_avg_mem_usage = bytes_to_mib(average_90_percent(l_memory_capacity))
            print(",".join(["GPT2", str(batch_size), str_avg_inf_time, str_avg_mem_usage]))
        '''
    else:
        print("Unidentified model name: {}".format(model_name))
        return

if __name__ == "__main__":
    main()

