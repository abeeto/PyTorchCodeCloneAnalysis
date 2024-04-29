import PIL.Image
import streamlit as st
import torch
from timm import create_model
from torchcam.methods import (
    CAM,
    GradCAM,
    GradCAMpp,
    LayerCAM,
    ScoreCAM,
    SmoothGradCAMpp,
    XGradCAM,
)
from torchcam.methods._utils import locate_candidate_layer
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import normalize, to_pil_image, to_tensor

MODEL_LIST = [
    "cspresnext50-0.894",
    "densenet121-0.900",
    "densenet169-0.899",
    "mobilenetv3-0.888",
]
CAM_METHODS = {
    "CAM": CAM,
    "GradCAM": GradCAM,
    "GradCAMpp": GradCAMpp,
    "SmoothGradCAMpp": SmoothGradCAMpp,
    "ScoreCAM": ScoreCAM,
    "XGradCAM": XGradCAM,
    "LayerCAM": LayerCAM,
}

st.set_page_config(page_title="CAM", layout="wide")
st.title("TorchCAM for Fracture Detection")
st.write("\n")
cols = st.columns(2)
cols[0].header("Input Image")
cols[1].header("Overlayed Image")

# sidebar
# image upload
st.sidebar.title("Input Selection")
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png"])
if uploaded_file is not None:
    img = PIL.Image.open(uploaded_file).convert("L")
    img = img.resize((512, 512), PIL.Image.Resampling.BICUBIC)
    cols[0].image(img, use_column_width=True)

# model
selected_model = st.sidebar.selectbox("Model", MODEL_LIST, index=0)

with st.spinner("Loading Model..."):
    if selected_model.startswith("densenet121"):
        model = create_model("densenet121", in_chans=1, num_classes=2)
    elif selected_model.startswith("densenet169"):
        model = create_model("densenet169", in_chans=1, num_classes=2)
    elif selected_model.startswith("mobilenetv3"):
        model = create_model("mobilenetv3_large_100", in_chans=1, num_classes=2)
    elif selected_model.startswith("convnext_tiny"):
        model = create_model("convnext_tiny", in_chans=1, num_classes=2)
    elif selected_model.startswith("cspresnext50"):
        model = create_model("cspresnext50", in_chans=1, num_classes=2)
    else:
        raise ValueError("Unknown model")

    state = torch.load(f"model/{selected_model}.pth")
    model.load_state_dict(state)

# layer
if selected_model.startswith("densenet121"):
    default_layer = "features"
elif selected_model.startswith("mobilenetv3"):
    default_layer = "blocks"
else:
    default_layer = locate_candidate_layer(model, (1, 512, 512))
target_layer = st.sidebar.text_input("Target Layer", default_layer)

# method
cam_method = st.sidebar.selectbox("CAM Method", CAM_METHODS.keys(), index=6)
cam_class = CAM_METHODS[cam_method]
cam_extractor = cam_class(
    model,
    target_layer=[s.strip() for s in target_layer.split("+")]
    if len(target_layer) > 0
    else None,
)

# alpha
alpha = st.sidebar.slider("Alpha", 0.0, 1.0, 0.7, step=0.05)

st.sidebar.write("\n")

# button
if st.sidebar.button("Compute CAM"):
    if uploaded_file is None:
        st.sidebar.error("Please upload an image first")
        st.stop()
    with st.spinner("Analyzing..."):
        inp = to_tensor(img)
        inp = normalize(inp, mean=[0.445], std=[0.269])

        out = model(inp.unsqueeze(0))

    prob = torch.softmax(out, dim=1).squeeze()[1].item()

    act_maps = cam_extractor(1, out)
    act_map = act_maps[0] if len(act_maps) == 1 else cam_extractor.fuse_cams(act_maps)

    act_map_pil = to_pil_image(act_map, mode="F")
    result = overlay_mask(img.convert("RGB"), act_map_pil, alpha=alpha)

    cols[1].image(result, use_column_width=True)
    cols[1].markdown(f"#### Probability: {prob:.3f}")
