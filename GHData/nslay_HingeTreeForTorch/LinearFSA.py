# 
# Nathan Lay
# AI Resource at National Cancer Institute
# National Institutes of Health
# December 2020
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
import torch.nn.functional as F

# Find FSA's details here:
# Barbu, Adrian, et al. "Feature selection with annealing for computer vision and big data learning." IEEE transactions on pattern analysis and machine intelligence 39.2 (2016): 272-286.

class LinearFSA(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(LinearFSA, self).__init__(*args, **kwargs) # XXX: This calls reset_parameters()
        # selection shape: out_features x in_features
        self.selection = nn.Parameter(torch.ones_like(self.weight), requires_grad=False) # NOTE: This must be assigned after nn.Module's constructor runs!

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.selection * self.weight, self.bias)

    def reset_parameters(self) -> None:
        super(LinearFSA, self).reset_parameters()
        
        if hasattr(self, "selection"): # XXX: Sanity check for call from __init__()
            self.selection[:] = 1
        
    def select(self, keep: int) -> None:
        if keep >= self.in_features:
            return
        
        if keep <= 0: # Why do this?
            self.selection[:] = 0
            return
            
        k = self.in_features - keep
        
        thresholds = (self.selection * self.weight).abs().kthvalue(k=k, dim=1, keepdim=True)[0]
        
        self.selection[self.weight.abs() <= thresholds] = 0
    
    # NOTE: Can feed in p = float("inf") for infinity norm too...
    def group_select(self, keep: int, p = "fro") -> None:
        if keep >= self.in_features:
            return
            
        if keep <= 0: # Why do this?
            self.selection[:] = 0
            return
            
        k = self.in_features - keep
        
        values = (self.selection * self.weight).norm(dim=0, p=p)
        threshold = values.kthvalue(k=k)[0]
        
        self.selection[:, values <= threshold] = 0
        
