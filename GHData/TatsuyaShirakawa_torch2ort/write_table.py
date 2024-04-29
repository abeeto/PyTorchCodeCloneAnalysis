import pandas as pd
from pytablewriter import MarkdownTableWriter

data = pd.read_json('result/results.json').T[['export to onnx',
                                              'pytorch loading',
                                              'onnxruntime loading',
                                              'pytorch inference',
                                              'onnxruntime inference',
                                              'inference speedup']]
writer = MarkdownTableWriter()
writer.from_dataframe(data, add_index_column=True)
writer.write_table()
