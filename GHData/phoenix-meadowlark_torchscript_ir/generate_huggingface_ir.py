import argparse
import os
import textwrap

import utils
import torch
import transformers


def _is_related_to_tf(cls):
  # Hacky, but means we don't need to install tensorflow to check for
  # tf.keras.layer subclassing.
  return "modeling_tf_" in str(cls)


def _get_model_name(model_class):
  model_name = str(model_class)
  model_name = model_name.replace("<class '", "").replace("'>", "")
  model_name = model_name.split(".")[-1]
  return model_name


CONFIG_TO_MODELS = {}
for attr in dir(transformers):
  value = getattr(transformers, attr)
  if hasattr(value, "config_class") and not _is_related_to_tf(value):
    config_class = value.config_class
    if config_class not in CONFIG_TO_MODELS:
      CONFIG_TO_MODELS[config_class] = []
    CONFIG_TO_MODELS[config_class].append(value)

ALL_MODELS = []
for _, models in CONFIG_TO_MODELS.items():
  ALL_MODELS.extend([_get_model_name(model) for model in models])
# PreTrainedModels wont work without loading weights.
ALL_MODELS = [
    model for model in ALL_MODELS if "pretrainedmodel" not in model.lower()
]

MODEL_TO_KWARGS = {
    "MBartForConditionalGeneration": {
        "num_beams": 8
    }
}


def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--models",
      default="all",
      help=f"Either 'all' or a comma delimited list of {ALL_MODELS}.")
  parser.add_argument("--save_models",
                      action="store_true",
                      help="Whether or not to save the compiled model.")
  parser.add_argument("--recompile",
                      action="store_true",
                      help="Whether or not to recompile the model if it is "
                      "already on disk.")
  return parser.parse_args()


def main():
  args = parse_arguments()
  model_names = ALL_MODELS if args.models == "all" else args.models.split(",")
  assert all(model in ALL_MODELS for model in model_names)

  for config_class, models in CONFIG_TO_MODELS.items():
    for model_class in models:

      model_name = str(model_class)
      model_name = model_name.replace("<class '", "").replace("'>", "")
      model_name = model_name.split(".")[-1]
      if model_name not in model_names:
        continue

      model_path = os.path.join("tsir", "huggingface", model_name)
      if os.path.exists(model_path) and not args.recompile:
        continue

      print(model_name)
      if model_name in MODEL_TO_KWARGS:
        kwargs = MODEL_TO_KWARGS[model_name]
      else:
        kwargs = {}
      model = None
      try:
        # Based off of the conversion in:
        #   https://huggingface.co/transformers/torchscript.html

        # Sometimes fails because of missing encoder / decoder specification.
        config = config_class(torchscript=True, **kwargs)
        model = model_class(config)
        model.eval()
        # batch_size = 17, num_choices = 7, sequence_length = 13
        # Chose these since random dimensions in the trace are unlikely to be
        # primes.
        if ("batch_size, num_choices, sequence_length" in model.forward.__doc__
            or "MultipleChoice" in model_name):
          inputs = [
              torch.zeros(17, 7, 13, dtype=torch.int64),
              torch.zeros(17, 7, 13, dtype=torch.int64)
          ]
        else:
          inputs = [
              torch.zeros(17, 13, dtype=torch.int64),
              torch.zeros(17, 13, dtype=torch.int64)
          ]
        jit_model = torch.jit.trace(model, inputs)
        utils.save_tsir(model, jit_model, model_path, args.save_models)
      except Exception as e:
        print()
        print(textwrap.indent(str(e), "    "))
        print()
      del model


if __name__ == "__main__":
  main()
