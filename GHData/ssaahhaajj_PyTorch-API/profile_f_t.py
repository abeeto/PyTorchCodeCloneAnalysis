from thop import profile as pf_flop
import torchprof as pf_time
from pandas import DataFrame
import pandas as pd

def profile(model, inp_data, want_op_file=False, cuda_=False):
  df1 = pf_flop(model, inputs=(inp_data, ))  
  with pf_time.Profile(model, use_cuda=cuda_) as prof:
    model(inp_data)
  df2=prof.display()
  for i1 in df1.index:
    df1["Layer_Name"][i1]=df2["Layer_Name"][i1]
  #   print(df1)
  #   print(df2)
  #   mynn={"Layer Name":[],"FLOPs":[],"Self CPU total":[], "CPU Total":[], "GPU Total":[],"Input Features":[], "Output Features":[], "Dict Size of Emb":[], "Emb Vector Size":[], "Norm Size":[]}
  #   for i1 in df1.index:
  #     mynn["Layer Name"].append(str(df2["Layer Name"][i1]))
  #     mynn["Self CPU total"].append(str(df2["Self CPU total"][i1]))
  #     mynn["CPU Total"].append(str(df2["CPU total"][i1]))
  #     mynn["GPU Total"].append(str(df2["GPU total"][i1]))
  #     mynn["Input Features"].append(str(df1["Input Features"][i1]))
  #     mynn["Output Features"].append(str(df1["Output Features"][i1]))
  #     mynn["Dict Size of Emb"].append(str(df1["Dict Size of Emb"][i1]))
  #     mynn["Emb Vector Size"].append(str(df1["Emb Vector Size"][i1]))
  #     mynn["Norm Size"].append(str(df1["Norm Size"][i1]))

  #   df=DataFrame(mynn, columns= ["Layer Name","FLOPs","Self CPU total","CPU Total","GPU Total","Input Features","Output Features","Dict Size of Emb","Emb Vector Size","Norm Size"])
  del df2["Layer_Name"]
  df = pd.concat([df1, df2], axis=1).reindex(df1.index)
  if want_op_file==True:
    export_csv = df.to_csv (r'output_file.csv', index = None, header=True)
  else:
    print(df)    
    
