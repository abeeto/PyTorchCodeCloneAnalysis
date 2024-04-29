from openvino.inference_engine import IENetwork, IEPlugin
import numpy as np

from config import lr, model_name, ckpt_name
model_xml = ckpt_name+'.xml'
model_bin = ckpt_name+'.bin'
test_dataset = 'data/test_data.npz'

def load_model(device, model_xml, model_bin):
  plugin = IEPlugin(device=device, plugin_dirs=None)
  net = IENetwork(model=model_xml, weights=model_bin)
  exec_net = plugin.load(network=net)
  input_blob = next(iter(net.inputs))
  output_blob = next(iter(net.outputs))
  return exec_net, input_blob, output_blob

def load_input(datafile):
  f = np.load(datafile)
  return f['images'], f['labels']

def print_perf_counts(exec_net):
  perf_counts = exec_net.requests[0].get_perf_counts()
  print("{:<70} {:<15} {:<15} {:<15} {:<10}".format('name', 'layer_type', 'exet_type', 'status', 'real_time, us'))
  for layer, stats in perf_counts.items():
    print("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer, stats['layer_type'], stats['exec_type'], stats['status'], stats['real_time']))

def run_inference(exec_net, input_blob, output_blob, images, labels):
  data_counts = images.shape[0]
  hit_counts = 0
  for i in range(data_counts):
    res = exec_net.infer(inputs={input_blob: images[i]})
    pred = res[output_blob].argmax()
    if i == 0:
      print(images[i])
      print(res, pred, labels[i])
    if pred == labels[i]:
      hit_counts += 1 
  accuracy = float(hit_counts)/float(data_counts)
  return accuracy

def main():
  # load model
  exec_net, input_blob, output_blob = load_model('CPU', model_xml, model_bin)
  # load input
  images, labels = load_input(test_dataset)
  # run inference
  accuracy = run_inference(exec_net, input_blob, output_blob, images, labels)
  print('accuracy = {}'.format(accuracy))

if __name__=='__main__':
  main()

