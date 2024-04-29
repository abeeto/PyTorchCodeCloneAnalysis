import os
from typing import Optional, Tuple
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from PIL import Image
from transformers import ViTFeatureExtractor, GPT2Tokenizer, BeitFeatureExtractor, DeiTFeatureExtractor, \
    RobertaTokenizer
import numpy as np
from transformers.configuration_utils import PretrainedConfig
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModel, AutoModelForCausalLM
from transformers.models.vision_encoder_decoder.configuration_vision_encoder_decoder import VisionEncoderDecoderConfig
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import Seq2SeqLMOutput

class generate_adaptor():
    def __init__(self, CheckpointPath:str=None):
        if CheckpointPath is None:
            raise ValueError("You have to specify Checkpoint Path")
        self.CheckpointPath = CheckpointPath
        head_tail = os.path.split(CheckpointPath)
        try:
            self.model = VisionEncoderDecoderModel.from_pretrained(head_tail[0], training=False)
            self.model.training = False
            self.model.config.bos_token_id = self.model.decoder.config.eos_token_id
            self.model.config.eos_token_id = self.model.decoder.config.eos_token_id
            self.model.config.decoder_start_token_id = self.model.config.eos_token_id
            self.model.config.pad_token_id = self.model.config.eos_token_id
            self.EncoderName = self.model.config.encoder.name_or_path
            self.DecoderName = self.model.config.decoder.name_or_path
            if "vit" in self.EncoderName:
                self.FeatureExtractor = ViTFeatureExtractor.from_pretrained(self.EncoderName)
            if "beit" in self.EncoderName:
                self.FeatureExtractor = BeitFeatureExtractor.from_pretrained(self.EncoderName)
            if "deit" in self.EncoderName:
                self.FeatureExtractor = DeiTFeatureExtractor.from_pretrained(self.EncoderName)
            if "roberta" in self.DecoderName:
                self.Tokenizer = RobertaTokenizer.from_pretrained(self.DecoderName)
            if "gpt2" in self.DecoderName:
                self.Tokenizer = GPT2Tokenizer.from_pretrained(self.DecoderName)
            if self.FeatureExtractor is None or self.Tokenizer is None:
                raise ValueError("please add a case above for the new model tokenizer and avoid fast tokenizers ")
            self.Tokenizer.eos_token = self.model.config.eos_token_id
            self.Tokenizer.pad_token = self.model.config.pad_token_id
            self.Tokenizer.add_special_tokens = True
        except:
            raise ValueError(
                "No configuration found on this folder please copy the necessary files to the checkpoint folder")

    def generate(self, Full_Image:Image=None, Crops:[]=None):
        from collections import OrderedDict as OD

        if Crops is None or Full_Image is None:
            raise ValueError("Crop and or Full Image must not be None")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        pixel_values_crops = []

        if isinstance(Full_Image, list) or isinstance(Full_Image, np.ndarray):
            Full_Image.append(Image.fromarray(Full_Image).convert('YCbCr').getchannel(0).convert('RGB'))

        if all(isinstance(n, list) or isinstance(n, np.ndarray) for n in Crops):
            for ImageCrop in Crops:
                try:
                    pixel_values_crops.append(Image.fromarray(ImageCrop).convert('YCbCr').getchannel(0).convert('RGB'))
                except:
                    print("Error converting array to image check image size of array != empty")

        Images = self.FeatureExtractor(images=[Full_Image, *pixel_values_crops], return_tensors="pt")
        pixel_values = Images["pixel_values"].reshape(1, 3 * Images["pixel_values"].shape[0], self.FeatureExtractor.size, self.FeatureExtractor.size)
        pixel_values = pixel_values.to(device)

        checkpoint = torch.load(self.CheckpointPath)

        model_state_dict = checkpoint['model_state_dict']

        encoder_checkpoint = {argument: value for argument, value in model_state_dict.items() if
                              argument.startswith("encoder.")}

        enc_to_dec_proj_checkpoint = {argument: value for argument, value in model_state_dict.items() if
                              argument.startswith("enc_to_dec_proj")}

        decoder_checkpoint = {argument[0:len("decoder")] + argument[len("decoder.decoder_name"):]: value for
                                   argument, value in model_state_dict.items() if argument.startswith("decoder.decoder_name")}

        if len(encoder_checkpoint) > 0 and len(decoder_checkpoint) > 0:
            encoder_decoder = OD()
            encoder_decoder.update(encoder_checkpoint)
            if len(enc_to_dec_proj_checkpoint) > 0:
                encoder_decoder.update(enc_to_dec_proj_checkpoint)
            encoder_decoder.update(decoder_checkpoint)
        else:
            raise ValueError("Wrong CheckPoint train using the custom module")

        self.model.load_state_dict(encoder_decoder)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        name_output = self.model.generate(pixel_values=pixel_values)

        decoder_checkpoint = {argument[0:len("decoder")] + argument[len("decoder.decoder_location"):]: value for
                                       argument, value in model_state_dict.items() if argument.startswith("decoder.decoder_location")}

        if len(encoder_checkpoint) > 0 and len(decoder_checkpoint) > 0:
            encoder_decoder = OD()
            encoder_decoder.update(encoder_checkpoint)
            if len(enc_to_dec_proj_checkpoint) > 0:
                encoder_decoder.update(enc_to_dec_proj_checkpoint)
            encoder_decoder.update(decoder_checkpoint)
        else:
            raise ValueError("Wrong CheckPoint train using the custom module")

        self.model.load_state_dict(encoder_decoder)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        location_output = self.model.generate(pixel_values=pixel_values)

        decoder_checkpoint = {argument[0:len("decoder")] + argument[len("decoder.decoder_date"):]: value for
                                   argument, value in model_state_dict.items() if argument.startswith("decoder.decoder_date")}

        if len(encoder_checkpoint) > 0 and len(decoder_checkpoint) > 0:
            encoder_decoder = OD()
            encoder_decoder.update(encoder_checkpoint)
            if len(enc_to_dec_proj_checkpoint) > 0:
                encoder_decoder.update(enc_to_dec_proj_checkpoint)
            encoder_decoder.update(decoder_checkpoint)
        else:
            raise ValueError("Wrong CheckPoint train using the custom module")

        self.model.load_state_dict(encoder_decoder)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        date_output = self.model.generate(pixel_values=pixel_values)

        def decode(token_ids, tokenizer, skip_special_tokens=True, clean_up_tokenization_spaces=True,
                   spaces_between_special_tokens=True):
            Output_text = []
            for token_id in token_ids:
                filtered_tokens = tokenizer.convert_ids_to_tokens(token_id, skip_special_tokens=skip_special_tokens)
                sub_texts = []
                current_sub_text = []
                for token in filtered_tokens:
                    if skip_special_tokens and token in tokenizer.all_special_ids:
                        continue
                    if token in tokenizer.added_tokens_encoder:
                        if current_sub_text:
                            sub_texts.append(tokenizer.convert_tokens_to_string(current_sub_text))
                            current_sub_text = []
                        sub_texts.append(token)
                    else:
                        current_sub_text.append(token)
                if current_sub_text:
                    sub_texts.append(tokenizer.convert_tokens_to_string(current_sub_text))

                if spaces_between_special_tokens:
                    text = " ".join(sub_texts)
                else:
                    text = "".join(sub_texts)
                if clean_up_tokenization_spaces:
                    text = tokenizer.clean_up_tokenization(text)
                Output_text.append(text)
            return Output_text

        return decode(name_output, self.Tokenizer), decode(location_output, self.Tokenizer), decode(date_output, self.Tokenizer)

class CustomEncoders(ModelOutput):
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    hidden_states_mask: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    pooler_output: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    last_hidden_state: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


class Custom_encoders(torch.nn.Module):

    def combine_pixel_values(self,Sizes, FeatureExtractor, XFullImage, XCrop):
        #WordSize == featurextractor.size
        Images_p = torch.zeros((len(Sizes), max(Sizes) * 3, FeatureExtractor.size, FeatureExtractor.size), dtype=torch.float32)
        for i in range(len(Sizes)):
            Images = FeatureExtractor(images=[XFullImage[i], *XCrop[i][0:Sizes[i] - 1].tolist()], return_tensors="pt")
            Images_p[i][0:3 * Sizes[i]][:][:] = Images["pixel_values"].reshape(1, 3 * Sizes[i],  FeatureExtractor.size, FeatureExtractor.size)
        return Images_p
    #raise NotImplementedError("Check the logic implemented in the RunTrain.py.\n Start encoding the images and them vertically stacking the colour channels the model requires atleast 6 colour channels for its dual encoders (1 image per encoder)\n The first 3 Colour channels (a image) are used on a encoder, while the rest n multiple of 3 (multiple images) will be encoded by the second encoder \n The input Should be {Batch, max number of channels (when necessary padding with 0 for the next dimentions), FeatureExtration.encTokens.size , FeatureExtration.encTokens.size} ")

    def split_pixel_values(self, pixel_values, batch_lengths=None):
        return_value = dict()
        N_Images = int(pixel_values.shape[1] / 3)
        if N_Images < 2:
            raise ValueError("Not enough Pixel_Values this encoder needs atleast two (one for full image another for the crop or crops)")
        if batch_lengths is not None:
            if len(batch_lengths) != pixel_values.shape[0]:
                batch_lengths = None
                raise ValueError("in order to use Batch training with this model its required to pass a parameter to model called : encoder_batch_lengths={batch, lenghts_of_images}")
        return_value['Full_Image'] = torch.zeros((pixel_values.shape[0], 3, self.config.image_size, self.config.image_size), dtype=pixel_values.dtype, device=pixel_values.device)
        return_value['Crop'] = []
        if batch_lengths is None:
            batch_lengths = []
            for j in range(pixel_values.shape[0]):
                return_value['Full_Image'][j] = pixel_values[j][:3].unsqueeze(dim=0)
                Crop = []
                for i in range(1, N_Images):
                    if bool(torch.all(
                            torch.zeros((3, self.config.image_size, self.config.image_size), dtype=pixel_values.dtype, device=pixel_values.device).eq(
                                    pixel_values[j][i * 3:(i + 1) * 3]))):
                        break
                    else:
                        Crop.append(pixel_values[j][i * 3:(i + 1) * 3].unsqueeze(dim=0))
                return_value['Crop'].append(torch.cat(Crop, 0))
                batch_lengths.append(len(Crop) + 1)
        else:
            for j in range(pixel_values.shape[0]):
                return_value['Full_Image'][j] = pixel_values[j][:3].unsqueeze(dim=0)
                Crop = []
                for i in range(1, batch_lengths[j]):
                    Crop.append(pixel_values[j][i * 3:(i + 1) * 3].unsqueeze(dim=0))
                return_value['Crop'].append(torch.cat(Crop, 0))
        return_value['Full_Image'] = return_value['Full_Image'].to(pixel_values.device)
        return batch_lengths, return_value

    def __init__(self, model_args=None, config=None, name_or_path=None, kwargs_encoder=None):
        super().__init__()
        if config is None and (model_args is not None and name_or_path is not None):
            self.encoder_image = AutoModel.from_pretrained(name_or_path, *model_args,
                                                           **kwargs_encoder)
            self.encoder_label = AutoModel.from_pretrained(name_or_path, *model_args,
                                                           **kwargs_encoder)
            self.update_config(self.encoder_label.config)
            self.config_class = self.encoder_label.config_class
            self.base_model_prefix = self.encoder_label.base_model_prefix
            self.main_input_name = self.encoder_label.main_input_name
            self.supports_gradient_checkpointing = self.encoder_label.supports_gradient_checkpointing
        elif config is not None and (model_args is None and name_or_path is None):
            self.config = config
            self.encoder_image = AutoModel.from_config(config)
            self.encoder_label = AutoModel.from_config(config)
            self.config_class = self.encoder_label.config_class
            self.base_model_prefix = self.encoder_label.base_model_prefix
            self.main_input_name = self.encoder_label.main_input_name
            self.supports_gradient_checkpointing = self.encoder_label.supports_gradient_checkpointing
        else:
            raise ValueError("Not enough parameters for a encoder configuration")

    def update_config(self, config):
        self.config = config
        self.encoder_image.config = config
        self.encoder_label.config = config

    def get_input_embeddings(self):
        return self.encoder_image.get_input_embeddings()

    def get_output_embeddings(self):
        return None

    def reshape_Tensor(self, input_value):
        sizes = []
        for i in range(len(input_value)):
            sizes.append(input_value[i].shape[0])
        output_value = []
        for i in range(len(input_value)):
            mid_value = []
            for j in range(input_value[i].shape[0]):
                mid_value.append(input_value[i][j])
            for j in range(max(sizes) - input_value[i].shape[0]):
                mid_value.append(torch.zeros(input_value[i][0].shape, dtype=input_value[i].dtype, device=input_value[i].device))
            output_value.append(torch.cat(mid_value, dim=0).unsqueeze(0))
        return torch.cat(output_value, dim=0)

    def forward(self, pixel_values, output_attentions, output_hidden_states, return_dict, **kwargs_encoder):
        try:
            batch_lengths = kwargs_encoder.pop("encoder_batch_lengths")
        except:
            batch_lengths = None

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values for both encoders")
        try:
            batch_lengths, pixel_values = self.split_pixel_values(pixel_values, batch_lengths)
        except:
            raise ValueError("Use encoder.combine_pixel_values to combine both image feature extration for this model")

        if pixel_values['Full_Image'] is None or pixel_values['Crop'] is None:
            raise ValueError("You have to specify pixel_values for both encoders")

        if pixel_values['Full_Image'].shape[0] != len(pixel_values['Crop']):
            raise ValueError("There was a problem with the split_pixel_values there must be the same full images as crop arrays")

        encoder_outputs_image = self.encoder_image(
            pixel_values['Full_Image'],
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs_encoder,
        )

        if self.config.output_hidden_states:
            encoder_hidden_states = []
            for jj in range(len(batch_lengths)):
                aux_hidden_states = []
                for hidden_state in encoder_outputs_image.hidden_states:
                    aux_hidden_states.append(hidden_state[jj:jj + 1])
                encoder_hidden_states.append([aux_hidden_states.copy()])
        else:
            encoder_hidden_states = None

        if self.config.output_attentions:
            encoder_attentions = []
            for jj in range(len(batch_lengths)):
                aux_attentions = []
                for attention in encoder_outputs_image.attentions:
                    aux_attentions.append(attention[jj:jj + 1])
                encoder_attentions.append([aux_attentions.copy()])
        else:
            encoder_attentions = None

        encoder_last_hidden_state = []
        encoder_pooler_output = []

        for i in range(len(pixel_values['Crop'])):
            encoder_outputs_label = self.encoder_label(
                pixel_values['Crop'][i],
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
            if encoder_attentions:
                encoder_attentions[i].append(encoder_outputs_label.attentions)

            if encoder_hidden_states:
                encoder_hidden_states[i].append(encoder_outputs_image.hidden_states)

            encoder_last_hidden_state.append(torch.cat((encoder_outputs_image[0][i:i + 1], encoder_outputs_label[0]), dim=0))
            encoder_pooler_output.append(torch.cat((encoder_outputs_image.pooler_output[i:i + 1], encoder_outputs_label.pooler_output), dim=0))


        encoder_pooler_output = self.reshape_Tensor(encoder_pooler_output)
        encoder_last_hidden_state = self.reshape_Tensor(encoder_last_hidden_state)

        if len(batch_lengths) > 1:
            last_hidden_state_mask = torch.zeros(encoder_last_hidden_state.shape[:2], dtype=torch.int16, device=encoder_last_hidden_state.device)
            for i in range(last_hidden_state_mask.shape[0]):
                last_hidden_state_mask[i][:int(batch_lengths[i] * encoder_outputs_image[0].shape[1])] = torch.ones(int(batch_lengths[i] * encoder_outputs_image[0].shape[1]), dtype=torch.int16, device=encoder_last_hidden_state.device)
        else:
            last_hidden_state_mask = None

        return CustomEncoders(attentions=encoder_attentions,
                            hidden_states=encoder_hidden_states,
                            pooler_output=encoder_pooler_output,
                            last_hidden_state_mask=last_hidden_state_mask,
                            last_hidden_state=encoder_last_hidden_state
                            )


class CustomDecoders(ModelOutput):
    logits_name: torch.FloatTensor = None
    logits_location: torch.FloatTensor = None
    logits_date: torch.FloatTensor = None
    past_key_values_name: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values_location: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values_date: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states_name: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states_location: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states_date: Optional[Tuple[torch.FloatTensor]] = None
    attentions_name: Optional[Tuple[torch.FloatTensor]] = None
    attentions_location: Optional[Tuple[torch.FloatTensor]] = None
    attentions_date: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions_name: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions_location: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions_date: Optional[Tuple[torch.FloatTensor]] = None
    output_name: Optional[Tuple[torch.FloatTensor]] = None,
    output_location: Optional[Tuple[torch.FloatTensor]] = None,
    output_date: Optional[Tuple[torch.FloatTensor]] = None


class Custom_decoders(torch.nn.Module):

    def combine_labels(self,Tokenizer ,YName, YLocation, YDate):
        return Tokenizer([*YName, *YLocation, *YDate], padding=True, truncation=True, max_length=512, return_tensors="pt")

    def __init__(self, config=None, name_or_path=None, kwargs_decoder=None):
        super().__init__()
        if config is None and name_or_path is not None:
            self.decoder_name = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs_decoder)
            self.decoder_location = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs_decoder)
            self.decoder_date = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs_decoder)
            self.update_config(self.decoder_date.config)
        elif config is not None and name_or_path is None:
            self.config = config
            self.decoder_name = AutoModelForCausalLM.from_config(config)
            self.decoder_location = AutoModelForCausalLM.from_config(config)
            self.decoder_date = AutoModelForCausalLM.from_config(config)
        else:
            raise ValueError("Not enough  raise parameters for a decoder configuration")


    def _reorder_cache(self, past, beam_idx):
        return self.decoder_name._reorder_cache(past, beam_idx), self.decoder_location._reorder_cache(past, beam_idx), self.decoder_date._reorder_cache(past, beam_idx)

    def get_output_embeddings(self):
        return self.decoder_name.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.decoder_name.set_output_embeddings(new_embeddings), self.decoder_location.set_output_embeddings(
            new_embeddings), self.decoder_date.set_output_embeddings(new_embeddings)

    def update_config(self, config):
        self.config = config
        self.decoder_name.config = config
        self.decoder_location.config = config
        self.decoder_date.config = config

    def split_decoder_input_withoutmask(self, InputVal):
        if InputVal.shape[0] % 3 != 0 and InputVal.shape[0] < 3:
            raise ValueError("The depth of the attention_mask must be the same as the input_ids and both depth must be  multiple of 3 to match: Name..., Locality..., Date...")
        batch = int(InputVal.shape[0] / 3)
        OutputVal = dict()
        OutputVal['Name'] = InputVal[0 * batch:1 * batch]
        OutputVal['Location'] = InputVal[1 * batch:2 * batch]
        OutputVal['Date'] = InputVal[2 * batch:3 * batch]
        return OutputVal

    def split_decoder_input_withmask(self, in_input_ids, in_attention_mask):
        if in_input_ids.shape[0] % 3 != 0 and in_attention_mask.shape[0] % 3 != 0 and in_input_ids.shape[0] != in_attention_mask.shape[0]:
            raise ValueError("The depth of the attention_mask must be the same as the input_ids and both depth must be multiple of 3 to match: Name..., Locality..., Date...")
        batch = int(in_input_ids.shape[0] / 3)
        if batch >= 2:
            input_ids = self.split_decoder_input_withoutmask(in_input_ids)
            attention_mask = self.split_decoder_input_withoutmask(in_attention_mask)
        else:
            input_ids = dict()
            attention_mask = dict()
            sizes = []
            for i in range(in_attention_mask.shape[0]):
                sizes.append(int(sum(in_attention_mask[i])))
            input_ids['Name'] = in_input_ids[0][:sizes[0]]
            input_ids['Name'] = torch.unsqueeze(input_ids['Name'], 0)
            input_ids['Location'] = in_input_ids[1][:sizes[1]]
            input_ids['Location'] = torch.unsqueeze(input_ids['Location'], 0)
            input_ids['Date'] = in_input_ids[2][:sizes[2]]
            input_ids['Date'] = torch.unsqueeze(input_ids['Date'], 0)
            attention_mask['Name'] = in_attention_mask[0][:sizes[0]]
            attention_mask['Name'] = torch.unsqueeze(attention_mask['Name'], 0)
            attention_mask['Location'] = in_attention_mask[1][:sizes[1]]
            attention_mask['Location'] = torch.unsqueeze(attention_mask['Location'], 0)
            attention_mask['Date'] = in_attention_mask[2][:sizes[2]]
            attention_mask['Date'] = torch.unsqueeze(attention_mask['Date'], 0)
        return input_ids, attention_mask

    def forward(self, input_ids, attention_mask, encoder_hidden_states, encoder_attention_mask, inputs_embeds, output_attentions, output_hidden_states, use_cache, past_key_values, return_dict, **kwargs_decoder):

        input_ids, attention_mask = self.split_decoder_input_withmask(input_ids, attention_mask)

        if inputs_embeds is None:
            inputs_embeds = dict()
            inputs_embeds['Name'] = None
            inputs_embeds['Location'] = None
            inputs_embeds['Date'] = None
        else:
            inputs_embeds = self.split_decoder_input_withoutmask(inputs_embeds)

        decoder_outputs_name = self.decoder_name(
            input_ids=input_ids['Name'],
            attention_mask=attention_mask['Name'],
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=inputs_embeds['Name'],
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        decoder_outputs_location = self.decoder_location(
            input_ids=input_ids['Location'],
            attention_mask=attention_mask['Location'],
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=inputs_embeds['Location'],
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        decoder_outputs_date = self.decoder_date(
            input_ids=input_ids['Date'],
            attention_mask=attention_mask['Date'],
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=inputs_embeds['Date'],
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        return CustomDecoders(
            logits_name=decoder_outputs_name.logits,
            logits_location=decoder_outputs_location.logits,
            logits_date=decoder_outputs_date.logits,
            past_key_values_date=decoder_outputs_date.past_key_values,
            past_key_values_name=decoder_outputs_name.past_key_values,
            past_key_values_location=decoder_outputs_location.past_key_values,
            hidden_states_date=decoder_outputs_date.hidden_states,
            hidden_states_name=decoder_outputs_name.hidden_states,
            hidden_states_location=decoder_outputs_location.hidden_states,
            attentions_date=decoder_outputs_date.attentions,
            attentions_name=decoder_outputs_name.attentions,
            attentions_location=decoder_outputs_location.attentions,
            cross_attentions_date=decoder_outputs_date.cross_attentions,
            cross_attentions_name=decoder_outputs_name.cross_attentions,
            cross_attentions_location=decoder_outputs_location.cross_attentions,
            output_name=decoder_outputs_name[0],
            output_location=decoder_outputs_location[0],
            output_date=decoder_outputs_date[0]
        )


class CustomSeq2SeqLMOutput(ModelOutput):
    loss_name: Optional[torch.FloatTensor] = None
    loss_location: Optional[torch.FloatTensor] = None
    loss_date: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    encoder_outputs: Optional[CustomEncoders] = None
    decoder_outputs: Optional[CustomDecoders] = None


# Copied from transformers.models.encoder_decoder.modeling_encoder_decoder.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    #shifted_input_ids = torch.tensor((), dtype=torch.float64)
    #shifted_input_ids.new_zeros(input_ids.shape)
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "VisionEncoderDecoderConfig"

VISION_ENCODER_DECODER_START_DOCSTRING = r"""
    This class can be used to initialize an image-to-text-sequence model with any pretrained vision autoencoding model
    as the encoder and any pretrained text autoregressive model as the decoder. The encoder is loaded via
    [`~AutoModel.from_pretrained`] function and the decoder is loaded via [`~AutoModelForCausalLM.from_pretrained`]
    function. Cross-attention layers are automatically added to the decoder and should be fine-tuned on a downstream
    generative task, like image captioning.

    The effectiveness of initializing sequence-to-sequence models with pretrained checkpoints for sequence generation
    tasks was shown in [Leveraging Pre-trained Checkpoints for Sequence Generation
    Tasks](https://arxiv.org/abs/1907.12461) by Sascha Rothe, Shashi Narayan, Aliaksei Severyn. Michael Matena, Yanqi
    Zhou, Wei Li, Peter J. Liu.

    Additionally, in [TrOCR: Transformer-based Optical Character Recognition with Pre-trained
    Models](https://arxiv.org/abs/2109.10282) it is shown how leveraging large pretrained vision models for optical
    character recognition (OCR) yields a significant performance improvement.

    After such a Vision-Encoder-Text-Decoder model has been trained/fine-tuned, it can be saved/loaded just like any
    other models (see the examples for more information).

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`VisionEncoderDecoderConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

VISION_ENCODER_DECODER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using a feature extractor (e.g. if you use ViT as the encoder,
            you should use [`ViTFeatureExtractor`]). See [`ViTFeatureExtractor.__call__`] for details.
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`PreTrainedTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            For training, `decoder_input_ids` are automatically created by the model by shifting the `labels` to the
            right, replacing -100 by the `pad_token_id` and prepending them with the `decoder_start_token_id`.
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        encoder_outputs (`tuple(torch.FloatTensor)`, *optional*):
            This tuple must consist of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) is a tensor
            of hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the
            decoder.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
            representation. This is useful if you want more control over how to convert `decoder_input_ids` indices
            into associated vectors than the model's internal embedding lookup matrix.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss for the decoder. Indices should be in `[-100, 0,
            ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            If set to `True`, the model will return a [`~file_utils.Seq2SeqLMOutput`] instead of a plain tuple.
        kwargs: (*optional*) Remaining dictionary of keyword arguments. Keyword arguments come in two flavors:

            - Without a prefix which will be input as `**encoder_kwargs` for the encoder forward function.
            - With a *decoder_* prefix which will be input as `**decoder_kwargs` for the decoder forward function.
"""


# @add_start_docstrings(VISION_ENCODER_DECODER_START_DOCSTRING) <-----------------------------------------
class VisionEncoderDecoderModel(PreTrainedModel):
    r"""
    [`VisionEncoderDecoderModel`] is a generic model class that will be instantiated as a transformer architecture with
    one of the base vision model classes of the library as encoder and another one as decoder when created with the
    :meth*~transformers.AutoModel.from_pretrained* class method for the encoder and
    :meth*~transformers.AutoModelForCausalLM.from_pretrained* class method for the decoder.
    """
    config_class = VisionEncoderDecoderConfig
    base_model_prefix = "vision_encoder_decoder"
    main_input_name = "pixel_values"
    training = bool

    def __init__(
            self,
            config: Optional[PretrainedConfig] = None,
            encoder: Optional[PreTrainedModel] = None,
            decoder: Optional[PreTrainedModel] = None,
            training: Optional[bool] = None
    ):
        if training is None:
            print("setting up training to True")
            training = True

        if config is None and (encoder is None or decoder is None):
            raise ValueError("Either a configuration or an encoder and a decoder has to be provided.")
        if config is None:
            config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        if config.decoder.cross_attention_hidden_size is not None:
            if config.decoder.cross_attention_hidden_size != config.encoder.hidden_size:
                raise ValueError(
                    "If `cross_attention_hidden_size` is specified in the decoder's configuration, "
                    "it has to be equal to the encoder's `hidden_size`. "
                    f"Got {config.decoder.cross_attention_hidden_size} for `config.decoder.cross_attention_hidden_size` "
                    f"and {config.encoder.hidden_size} for `config.encoder.hidden_size`."
                )

        # initialize with config
        # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False
        super().__init__(config)

        if encoder is None:
            encoder = Custom_encoders(config=config.encoder)

        if decoder is None:
            if training:
                decoder = Custom_decoders(config=config.decoder)
            else:
                decoder = AutoModelForCausalLM.from_config(config.decoder)

        self.encoder = encoder
        self.decoder = decoder

        if self.encoder.config.to_dict() != self.config.encoder.to_dict():
            logger.warning(
                f"Config of the encoder: {self.encoder.__class__} is overwritten by shared encoder config: {self.config.encoder}"
            )
        if self.decoder.config.to_dict() != self.config.decoder.to_dict():
            logger.warning(
                f"Config of the decoder: {self.decoder.__class__} is overwritten by shared decoder config: {self.config.decoder}"
            )

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.encoder.update_config(self.config.encoder)

        if training:
            self.decoder.update_config(self.config.decoder)
        else:
            self.decoder.config = self.config.decoder

        # encoder outputs might need to be projected to different dimension for decoder
        if (
                self.encoder.config.hidden_size != self.decoder.config.hidden_size
                and self.decoder.config.cross_attention_hidden_size is None
        ):
            self.enc_to_dec_proj = nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size)

        if self.encoder.get_output_embeddings() is not None:
            raise ValueError(
                f"The encoder {self.encoder} should not have a LM Head. Please use a model without LM Head"
            )

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported for composite models
        if kwargs.get("_fast_init", False):
            logger.warning(
                "Fast initialization is currently not supported for VisionEncoderDecoderModel. "
                "Falling back to slow initialization..."
            )
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    def from_encoder_decoder_pretrained(
            cls,
            encoder_pretrained_model_name_or_path: str = None,
            decoder_pretrained_model_name_or_path: str = None,
            training: bool = None,
            *model_args,
            **kwargs
    ) -> PreTrainedModel:
        r"""
        Instantiate an encoder and a decoder from one or two base classes of the library from pretrained model
        checkpoints.


        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you need to first set it back in training mode with `model.train()`.

        Params:
            encoder_pretrained_model_name_or_path (`str`, *optional*):
                Information necessary to initiate the image encoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co. An
                      example is `google/vit-base-patch16-224-in21k`.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            decoder_pretrained_model_name_or_path (`str`, *optional*, defaults to `None`):
                Information necessary to initiate the text decoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args (remaining positional arguments, *optional*):
                All remaning positional arguments will be passed to the underlying model's `__init__` method.

            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`).

                - To update the encoder configuration, use the prefix *encoder_* for each configuration parameter.
                - To update the decoder configuration, use the prefix *decoder_* for each configuration parameter.
                - To update the parent model configuration, do not use a prefix for each configuration parameter.

                Behaves differently depending on whether a `config` is provided or automatically loaded.

        Example:

        ```python
        >>> from transformers import VisionEncoderDecoderModel

        >>> # initialize a vit-bert from a pretrained ViT and a pretrained BERT model. Note that the cross-attention layers will be randomly initialized
        >>> model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        ...     "google/vit-base-patch16-224-in21k", "bert-base-uncased"
        ... )
        >>> # saving model after fine-tuning
        >>> model.save_pretrained("./vit-bert")
        >>> # load fine-tuned model
        >>> model = VisionEncoderDecoderModel.from_pretrained("./vit-bert")
        ```"""
        if training is None:
            print("setting up training to True")
            training = True
        kwargs_encoder = {
            argument[len("encoder_"):]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        # remove encoder, decoder kwargs from kwargs
        for key in kwargs_encoder.keys():
            del kwargs["encoder_" + key]
        for key in kwargs_decoder.keys():
            del kwargs["decoder_" + key]

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            if encoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `encoder_model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_encoder:
                encoder_config, kwargs_encoder = AutoConfig.from_pretrained(
                    encoder_pretrained_model_name_or_path, **kwargs_encoder, return_unused_kwargs=True
                )

                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                    logger.info(
                        f"Initializing {encoder_pretrained_model_name_or_path} as a encoder model "
                        "from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False

                kwargs_encoder["config"] = encoder_config

            encoder = Custom_encoders(name_or_path=encoder_pretrained_model_name_or_path, model_args=model_args,
                                      kwargs_encoder=kwargs_encoder)

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            if decoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_decoder:
                decoder_config, kwargs_decoder = AutoConfig.from_pretrained(
                    decoder_pretrained_model_name_or_path, **kwargs_decoder, return_unused_kwargs=True
                )

                if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
                    logger.info(
                        f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. "
                        f"Cross attention layers are added to {decoder_pretrained_model_name_or_path} "
                        f"and randomly initialized if {decoder_pretrained_model_name_or_path}'s architecture allows for "
                        "cross attention layers."
                    )
                    decoder_config.is_decoder = True
                    decoder_config.add_cross_attention = True

                kwargs_decoder["config"] = decoder_config

            if kwargs_decoder["config"].is_decoder is False or kwargs_decoder["config"].add_cross_attention is False:
                logger.warning(
                    f"Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. "
                    f"In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, "
                    "make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` "
                    "passed to `.from_encoder_decoder_pretrained(...)` are set to `True` or do not pass a "
                    "`decoder_config` to `.from_encoder_decoder_pretrained(...)`"
                )
            if training:
                decoder = Custom_decoders(name_or_path=decoder_pretrained_model_name_or_path,
                                          kwargs_decoder=kwargs_decoder)
            else:
                decoder = AutoModelForCausalLM.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)

        # instantiate config with corresponding kwargs
        config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config, **kwargs)

        # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False
        return cls(encoder=encoder, decoder=decoder, config=config, training=training)

    # @add_start_docstrings_to_model_forward(VISION_ENCODER_DECODER_INPUTS_DOCSTRING) <-----------------------------------------
    # @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC) <-----------------------------------------
    def forward(
            self,
            pixel_values=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            single_loss_sum=False,
            **kwargs,
    ):
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        >>> import requests
        >>> from PIL import Image
        >>> import torch

        >>> processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        >>> model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

        >>> # load image from the IAM dataset
        >>> url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

        >>> # training
        >>> model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
        >>> model.config.pad_token_id = processor.tokenizer.pad_token_id
        >>> model.config.vocab_size = model.config.decoder.vocab_size

        >>> pixel_values = processor(image, return_tensors="pt").pixel_values
        >>> text = "hello world"
        >>> labels = processor.tokenizer(text, return_tensors="pt").input_ids
        >>> outputs = model(pixel_values=pixel_values, labels=labels)
        >>> loss = outputs.loss

        >>> # inference (generation)
        >>> generated_ids = model.generate(pixel_values)
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None and CustomEncoders != type(encoder_outputs):
            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")

            encoder_outputs = self.encoder(
                pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )

        encoder_hidden_states = encoder_outputs.last_hidden_state
        encoder_attention_mask = encoder_outputs.last_hidden_state_mask

        if (self.encoder.config.hidden_size != self.decoder.config.hidden_size and self.decoder.config.cross_attention_hidden_size is None):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id,self.config.decoder_start_token_id)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        if self.training:
            loss = None
            labels, decoder_attention_mask, = self.decoder.split_decoder_input_withmask(labels, decoder_attention_mask)
            if labels["Name"] is not None and labels["Location"] is not None and labels["Date"] is not None:
                logits_location = decoder_outputs.logits_location if return_dict else decoder_outputs.output_location
                logits_name = decoder_outputs.logits_name if return_dict else decoder_outputs.output_name
                logits_date = decoder_outputs.logits_date if return_dict else decoder_outputs.output_date
                loss_fct = CrossEntropyLoss()
                if single_loss_sum:
                    loss_location = loss_fct(logits_location.reshape(-1, self.decoder.config.vocab_size), labels["Location"].view(-1))
                    loss_name = loss_fct(logits_name.reshape(-1, self.decoder.config.vocab_size), labels["Name"].view(-1))
                    loss_date = loss_fct(logits_date.reshape(-1, self.decoder.config.vocab_size), labels["Date"].view(-1))
                    loss = loss_location + loss_name + loss_date
                    if not return_dict:
                        if loss is not None:
                            return (loss,) + decoder_outputs + encoder_outputs
                        else:
                            return decoder_outputs + encoder_outputs
                    return CustomSeq2SeqLMOutput(loss=loss, loss_name=loss_name, loss_location=loss_location, loss_date=loss_date, encoder_outputs=encoder_outputs, decoder_outputs=decoder_outputs)
                else:
                    logits = torch.cat((logits_name, logits_location, logits_date), 1).reshape(-1, self.decoder.config.vocab_size)
                    expected = torch.cat((labels["Name"].view(-1), labels["Location"].view(-1), labels["Date"].view(-1)), 0)
                    loss = loss_fct(logits, expected)
                    if not return_dict:
                        if loss is not None:
                            return (loss,) + decoder_outputs + encoder_outputs
                        else:
                            return decoder_outputs + encoder_outputs
                    return CustomSeq2SeqLMOutput(loss=loss, encoder_outputs=encoder_outputs, decoder_outputs=decoder_outputs)
        else:
            # Compute loss independent from decoder (as some shift the logits inside them)
            loss = None
            if labels is not None:
                logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))

            if not return_dict:
                if loss is not None:
                    return (loss,) + decoder_outputs + encoder_outputs
                else:
                    return decoder_outputs + encoder_outputs

            return Seq2SeqLMOutput(
                loss=loss,
                logits=decoder_outputs.logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def prepare_inputs_for_generation(
            self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, past=past)
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
            "past_key_values": decoder_inputs["past_key_values"],
            "use_cache": use_cache,
        }
        return input_dict

    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError(
            "Resizing the embedding layers via the VisionEncoderDecoderModel directly is not supported."
            "Please use the respective methods of the wrapped decoder object (model.decoder.resize_token_embeddings(...))"
        )

    def _reorder_cache(self, past, beam_idx):
        # apply decoder cache reordering here
        return self.decoder._reorder_cache(past, beam_idx)

