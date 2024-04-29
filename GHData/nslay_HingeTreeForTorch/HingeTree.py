# 
# Nathan Lay
# AI Resource at National Cancer Institute
# National Institutes of Health
# November 2020
# 
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 

import torch
import torch.nn as nn
import torch.autograd
import hingetree_cpp

def _is_deterministic():
    if hasattr(torch, "are_deterministic_algorithms_enabled"):
        return torch.are_deterministic_algorithms_enabled()
    elif hasattr(torch, "is_deterministic"):
        return torch.is_deterministic()

    raise RuntimeError("Unable to query torch deterministic mode.")

def contract(inData, window, padding=0):
  if isinstance(window, int):
    window = [window]*(inData.dim()-2)

  if isinstance(padding, int):
    padding = [padding]*(inData.dim()-2)

  return hingetree_cpp.contract(inData, window, padding)

def expand(inData, padding=0):
  if isinstance(padding, int):
    padding = [padding]*((inData.dim()-2)//2)

  return hingetree_cpp.expand(inData, padding)

class HingeTree(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inData, inThresholds, inOrdinals, inWeights):
        ctx.save_for_backward(inData, inThresholds, inOrdinals, inWeights)

        return hingetree_cpp.tree_forward(inData, inThresholds, inOrdinals, inWeights)

    @staticmethod
    def backward(ctx, outDataGrad):
        inData, inThresholds, inOrdinals, inWeights = ctx.saved_tensors

        if _is_deterministic():
            inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad = hingetree_cpp.tree_backward_deterministic(inData, ctx.needs_input_grad[0], inThresholds, ctx.needs_input_grad[1], inOrdinals, ctx.needs_input_grad[2], inWeights, ctx.needs_input_grad[3], outDataGrad.contiguous())
        else:
            inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad = hingetree_cpp.tree_backward(inData, ctx.needs_input_grad[0], inThresholds, ctx.needs_input_grad[1], inOrdinals, ctx.needs_input_grad[2], inWeights, ctx.needs_input_grad[3], outDataGrad.contiguous())

        return inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad

    @staticmethod
    def check_thresholds(inThresholds, inOrdinals, inWeights):
        return hingetree_cpp.tree_check_thresholds(inThresholds, inOrdinals, inWeights)

    @staticmethod
    def fix_thresholds(inThresholds, inOrdinals, inWeights):
        return hingetree_cpp.tree_fix_thresholds(inThresholds, inOrdinals, inWeights)

    @staticmethod
    def reachability(inData, inThresholds, inOrdinals, inWeights):
        return hingetree_cpp.tree_reachability(inData, inThresholds, inOrdinals, inWeights)

    @staticmethod
    def leafmap(inData, inThresholds, inOrdinals, inWeights):
        return hingetree_cpp.tree_leafmap(inData, inThresholds, inOrdinals, inWeights)

    @staticmethod
    def marginmap(inData, inThresholds, inOrdinals, inWeights):
        return hingetree_cpp.tree_marginmap(inData, inThresholds, inOrdinals, inWeights)

    @staticmethod
    def speedtest(inData):
        return hingetree_cpp.tree_speedtest(inData, _is_deterministic())

    @staticmethod
    def init_medians(inData, inThresholds, inOrdinals, inWeights):
        return hingetree_cpp.tree_init_medians(inData, inThresholds, inOrdinals, inWeights)

    @staticmethod
    def init_greedy(inData, inLabels, inThresholds, inOrdinals, inWeights):
        return hingetree_cpp.tree_init_greedy(inData, inLabels, inThresholds, inOrdinals, inWeights)

class HingeFern(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inData, inThresholds, inOrdinals, inWeights):
        ctx.save_for_backward(inData, inThresholds, inOrdinals, inWeights)

        return hingetree_cpp.fern_forward(inData, inThresholds, inOrdinals, inWeights)

    @staticmethod
    def backward(ctx, outDataGrad):
        inData, inThresholds, inOrdinals, inWeights = ctx.saved_tensors

        if _is_deterministic():
            inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad = hingetree_cpp.fern_backward_deterministic(inData, ctx.needs_input_grad[0], inThresholds, ctx.needs_input_grad[1], inOrdinals, ctx.needs_input_grad[2], inWeights, ctx.needs_input_grad[3], outDataGrad.contiguous())
        else:
            inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad = hingetree_cpp.fern_backward(inData, ctx.needs_input_grad[0], inThresholds, ctx.needs_input_grad[1], inOrdinals, ctx.needs_input_grad[2], inWeights, ctx.needs_input_grad[3], outDataGrad.contiguous())

        return inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad
     
    @staticmethod
    def check_thresholds(inThresholds, inOrdinals, inWeights):
        return hingetree_cpp.fern_check_thresholds(inThresholds, inOrdinals, inWeights)

    @staticmethod
    def fix_thresholds(inThresholds, inOrdinals, inWeights):
        return hingetree_cpp.fern_fix_thresholds(inThresholds, inOrdinals, inWeights)

    @staticmethod
    def reachability(inData, inThresholds, inOrdinals, inWeights):
        return hingetree_cpp.fern_reachability(inData, inThresholds, inOrdinals, inWeights)

    @staticmethod
    def leafmap(inData, inThresholds, inOrdinals, inWeights):
        return hingetree_cpp.fern_leafmap(inData, inThresholds, inOrdinals, inWeights)

    @staticmethod
    def marginmap(inData, inThresholds, inOrdinals, inWeights):
        return hingetree_cpp.fern_marginmap(inData, inThresholds, inOrdinals, inWeights)

    @staticmethod
    def speedtest(inData):
        return hingetree_cpp.fern_speedtest(inData, _is_deterministic())

    @staticmethod
    def init_medians(inData, inThresholds, inOrdinals, inWeights):
        return hingetree_cpp.fern_init_medians(inData, inThresholds, inOrdinals, inWeights)

class HingeTreeFusedLinear(HingeTree):
    @staticmethod
    def forward(ctx, inData, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias):
        if _is_deterministic() and inData.device.type != "cpu":
            raise RuntimeError("No deterministic implementation of forward for hinge tree + linear on GPUs.")

        ctx.save_for_backward(inData, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias)

        return hingetree_cpp.tree_fused_linear_forward(inData, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias)

    @staticmethod
    def backward(ctx, outDataGrad):
        if _is_deterministic() and outDataGrad.device.type != "cpu":
            raise RuntimeError("No deterministic implementation of backpropagation for hinge tree + linear on GPUs.")

        inData, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias = ctx.saved_tensors

        inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad, inLinearWeightsGrad, inLinearBiasGrad = hingetree_cpp.tree_fused_linear_backward(inData, ctx.needs_input_grad[0], inThresholds, ctx.needs_input_grad[1], inOrdinals, ctx.needs_input_grad[2], inWeights, ctx.needs_input_grad[3], inLinearWeights, ctx.needs_input_grad[4], inLinearBias, ctx.needs_input_grad[5], outDataGrad.contiguous())

        return inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad, inLinearWeightsGrad, inLinearBiasGrad

# Convolution operations below

class HingeTreeConv1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inData, inThresholds, inOrdinals, inWeights, kernelSize, stride, padding, dilation):
        ctx.save_for_backward(inData, inThresholds, inOrdinals, inWeights)

        ctx.kernelSize = kernelSize
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation

        return hingetree_cpp.tree_conv1d_forward(inData, inThresholds, inOrdinals, inWeights, ctx.kernelSize, ctx.stride, ctx.padding, ctx.dilation)

    @staticmethod
    def backward(ctx, outDataGrad):
        if _is_deterministic() and outDataGrad.device.type != "cpu":
            raise RuntimeError("No deterministic implementation of backpropagation for hinge tree convolution on GPUs.")
    
        inData, inThresholds, inOrdinals, inWeights = ctx.saved_tensors

        inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad = hingetree_cpp.tree_conv1d_backward(inData, ctx.needs_input_grad[0], inThresholds, ctx.needs_input_grad[1], inOrdinals, ctx.needs_input_grad[2], inWeights, ctx.needs_input_grad[3], outDataGrad.contiguous(), ctx.kernelSize, ctx.stride, ctx.padding, ctx.dilation)

        return inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad, None, None, None, None

class HingeTreeConv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inData, inThresholds, inOrdinals, inWeights, kernelSize, stride, padding, dilation):
        ctx.save_for_backward(inData, inThresholds, inOrdinals, inWeights)

        ctx.kernelSize = kernelSize
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation

        return hingetree_cpp.tree_conv2d_forward(inData, inThresholds, inOrdinals, inWeights, ctx.kernelSize, ctx.stride, ctx.padding, ctx.dilation)

    @staticmethod
    def backward(ctx, outDataGrad):
        if _is_deterministic() and outDataGrad.device.type != "cpu":
            raise RuntimeError("No deterministic implementation of backpropagation for hinge tree convolution on GPUs.")
    
        inData, inThresholds, inOrdinals, inWeights = ctx.saved_tensors

        inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad = hingetree_cpp.tree_conv2d_backward(inData, ctx.needs_input_grad[0], inThresholds, ctx.needs_input_grad[1], inOrdinals, ctx.needs_input_grad[2], inWeights, ctx.needs_input_grad[3], outDataGrad.contiguous(), ctx.kernelSize, ctx.stride, ctx.padding, ctx.dilation)

        return inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad, None, None, None, None

class HingeTreeConv3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inData, inThresholds, inOrdinals, inWeights, kernelSize, stride, padding, dilation):
        ctx.save_for_backward(inData, inThresholds, inOrdinals, inWeights)

        ctx.kernelSize = kernelSize
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation

        return hingetree_cpp.tree_conv3d_forward(inData, inThresholds, inOrdinals, inWeights, ctx.kernelSize, ctx.stride, ctx.padding, ctx.dilation)

    @staticmethod
    def backward(ctx, outDataGrad):
        if _is_deterministic() and outDataGrad.device.type != "cpu":
            raise RuntimeError("No deterministic implementation of backpropagation for hinge tree convolution on GPUs.")
    
        inData, inThresholds, inOrdinals, inWeights = ctx.saved_tensors

        inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad = hingetree_cpp.tree_conv3d_backward(inData, ctx.needs_input_grad[0], inThresholds, ctx.needs_input_grad[1], inOrdinals, ctx.needs_input_grad[2], inWeights, ctx.needs_input_grad[3], outDataGrad.contiguous(), ctx.kernelSize, ctx.stride, ctx.padding, ctx.dilation)

        return inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad, None, None, None, None

class HingeFernConv1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inData, inThresholds, inOrdinals, inWeights, kernelSize, stride, padding, dilation):
        ctx.save_for_backward(inData, inThresholds, inOrdinals, inWeights)

        ctx.kernelSize = kernelSize
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation

        return hingetree_cpp.fern_conv1d_forward(inData, inThresholds, inOrdinals, inWeights, ctx.kernelSize, ctx.stride, ctx.padding, ctx.dilation)

    @staticmethod
    def backward(ctx, outDataGrad):
        if _is_deterministic() and outDataGrad.device.type != "cpu":
            raise RuntimeError("No deterministic implementation of backpropagation for hinge fern convolution on GPUs.")
    
        inData, inThresholds, inOrdinals, inWeights = ctx.saved_tensors

        inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad = hingetree_cpp.fern_conv1d_backward(inData, ctx.needs_input_grad[0], inThresholds, ctx.needs_input_grad[1], inOrdinals, ctx.needs_input_grad[2], inWeights, ctx.needs_input_grad[3], outDataGrad.contiguous(), ctx.kernelSize, ctx.stride, ctx.padding, ctx.dilation)

        return inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad, None, None, None, None

class HingeFernConv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inData, inThresholds, inOrdinals, inWeights, kernelSize, stride, padding, dilation):
        ctx.save_for_backward(inData, inThresholds, inOrdinals, inWeights)

        ctx.kernelSize = kernelSize
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation

        return hingetree_cpp.fern_conv2d_forward(inData, inThresholds, inOrdinals, inWeights, ctx.kernelSize, ctx.stride, ctx.padding, ctx.dilation)

    @staticmethod
    def backward(ctx, outDataGrad):
        if _is_deterministic() and outDataGrad.device.type != "cpu":
            raise RuntimeError("No deterministic implementation of backpropagation for hinge fern convolution on GPUs.")
        
        inData, inThresholds, inOrdinals, inWeights = ctx.saved_tensors
        
        inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad = hingetree_cpp.fern_conv2d_backward(inData, ctx.needs_input_grad[0], inThresholds, ctx.needs_input_grad[1], inOrdinals, ctx.needs_input_grad[2], inWeights, ctx.needs_input_grad[3], outDataGrad.contiguous(), ctx.kernelSize, ctx.stride, ctx.padding, ctx.dilation)

        return inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad, None, None, None, None

class HingeFernConv3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inData, inThresholds, inOrdinals, inWeights, kernelSize, stride, padding, dilation):
        ctx.save_for_backward(inData, inThresholds, inOrdinals, inWeights)

        ctx.kernelSize = kernelSize
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation

        return hingetree_cpp.fern_conv3d_forward(inData, inThresholds, inOrdinals, inWeights, ctx.kernelSize, ctx.stride, ctx.padding, ctx.dilation)

    @staticmethod
    def backward(ctx, outDataGrad):
        if _is_deterministic() and outDataGrad.device.type != "cpu":
            raise RuntimeError("No deterministic implementation of backpropagation for hinge fern convolution on GPUs.")
        
        inData, inThresholds, inOrdinals, inWeights = ctx.saved_tensors
        
        inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad = hingetree_cpp.fern_conv3d_backward(inData, ctx.needs_input_grad[0], inThresholds, ctx.needs_input_grad[1], inOrdinals, ctx.needs_input_grad[2], inWeights, ctx.needs_input_grad[3], outDataGrad.contiguous(), ctx.kernelSize, ctx.stride, ctx.padding, ctx.dilation)

        return inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad, None, None, None, None

class HingeTrie(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inData, inThresholds, inOrdinals, inWeights):
        ctx.save_for_backward(inData, inThresholds, inOrdinals, inWeights)

        return hingetree_cpp.trie_forward(inData, inThresholds, inOrdinals, inWeights)

    @staticmethod
    def backward(ctx, outDataGrad):
        if _is_deterministic() and outDataGrad.device.type != "cpu":
            raise RuntimeError("No deterministic implementation of backpropagation of hinge trie on GPUs.")
    
        inData, inThresholds, inOrdinals, inWeights = ctx.saved_tensors

        inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad = hingetree_cpp.trie_backward(inData, ctx.needs_input_grad[0], inThresholds, ctx.needs_input_grad[1], inOrdinals, ctx.needs_input_grad[2], inWeights, ctx.needs_input_grad[3], outDataGrad.contiguous())

        return inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad

    @staticmethod
    def init_medians(inData, inThresholds, inOrdinals, inWeights):
        return hingetree_cpp.trie_init_medians(inData, inThresholds, inOrdinals, inWeights)


