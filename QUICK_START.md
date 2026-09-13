# Quick Start

## 🧩 Basic Usage

### Running AutoGaze on a Video

Here's a simple script of running AutoGaze on a video.

```python
import av
import torch
from autogaze.datasets.video_utils import read_video_pyav, transform_video_for_pytorch
from autogaze.models.autogaze import AutoGazeImageProcessor, AutoGaze

# Load AutoGaze model and its preprocessor from HuggingFace
autogaze_transform = AutoGazeImageProcessor.from_pretrained("nvidia/AutoGaze")
autogaze_model = AutoGaze.from_pretrained("nvidia/AutoGaze")

# Load a video
video_path = "<path_to_autogaze_code>/assets/example_input.mp4"  # Fill in the path of AutoGaze codebase
container = av.open(video_path)
sample_indices = list(range(autogaze_model.config.max_num_frames))  # Sampling the first 16 frames (max_num_frames = 16).
raw_video = read_video_pyav(container=container, indices=sample_indices)
container.close()

# Preprocess the video
video_input_autogaze = transform_video_for_pytorch(raw_video, autogaze_transform)  # T * C * H * W
video_input_autogaze = video_input_autogaze[None]  # B * T * C * H * W

# Run the AutoGaze model
with torch.inference_mode():
    gaze_outputs = autogaze_model({"video": video_input_autogaze}, gazing_ratio=0.75, task_loss_requirement=0.7)  # Here we allow the gazing_ratio to be 0.75 (it can gaze at 75% of the patches) and set task_loss_requirement to 0.7 (it will stop gazing at each frame once the reconstruction loss falls under 0.7).
```

Something to note in this code:

- We resize the video to 224x224 and only load 16 frame from it since AutoGaze is only trained on 16-frame 224x224 videos. However, to process video with any resolution and any length, you can just chop it into 16-frame 224x224 chunks and run AutoGaze on every chunk separately, as we will briefly show later.

- When calling `autogaze_model`, there are two ways to control how many patches it gazes at for each frame, i.e., through `gazing_ratio` and `task_loss_requirement`. `gazing_ratio` controls the maximum percentage of patches it can gaze per frame, and `task_loss_requirement` sets a threshold of reconstruction loss such that the model will stop gazing at each frame once the patches already gazed can reach a reconstruction loss lower than the threshold. In our experiments, we find `gazing_ratio=0.75` and `task_loss_requirement=0.7` can usually maintain the downstream performance.


Now let's inspect the outputs:

```python
print(gaze_outputs['gazing_pos'].shape)  # 1 * 348. gazing_pos records the indices of the patches being gazed at. This means AutoGaze gazed at 348 patches (including padded gazing) for the video.
print(gaze_outputs['if_padded_gazing'].shape)  # 1 * 348. if_padded_gazing records which positions are padded (dummy) gazing. Each element of if_padded_gazing is boolean, and True means the gazing at that position is padded. 
print((~gaze_outputs['if_padded_gazing']).sum(dim=-1))  # 213. This means the actual number of gazed patches is 213.
print(gaze_outputs['num_gazing_each_frame'])  # [198, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]. This is the number of gazing per frame (including padding). 
```

Note:

- In `gaze_outputs['gazing_pos']`, each patch index is not counted from the first patch within its frame, but the first patch in the first frame. For example, if the 3rd patch in 5th frame is gazed at, its id in `gaze_outputs['gazing_pos']` is `(5 - 1) * 265 + 3 - 1 = 1327`, not `3 - 1 = 2`, assuming each frame has 265 patches and we count from 0.

### Applying AutoGaze to a Vision Encoder

We can apply AutoGaze to a vision encoder (e.g., SigLIP) and let the encoder only process the gazed patches to improve efficiency. To achieve this, we need to customize the vision encoder a bit to support encoding partially observed (multi-scale) patches. Here we've already implemented SigLIP to be compatible with AutoGaze. To make SigLIP only encode the gazed patches, all you need to do is to pass the `gaze_outputs` when calling it.

```python
from transformers import AutoImageProcessor
from autogaze.vision_encoders.siglip import SiglipVisionModel  # Our customized SigLIP to be compatible with AutoGaze

# Load SigLIP and its preprocessor
siglip_transform = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224")
siglip_model = SiglipVisionModel.from_pretrained("google/siglip2-base-patch16-224", scales=autogaze_model.config.scales, attn_implementation="sdpa")

# Preprocess the video with SigLIP preprocessor
video_input_siglip = transform_video_for_pytorch(raw_video, siglip_transform)  # T * C * H * W
video_input_siglip = video_input_siglip[None]  # B * T * C * H * W

# Encode the gazed patches with SigLIP
siglip_outputs = siglip_model(video_input_siglip, gazing_info=gaze_outputs)  # Here we pass in the gazing outputs such that SigLIP only encodes gazed patches. The customized SigLIP takes videos as input. 
print(siglip_outputs.last_hidden_state.shape)  # 1 * 348 * 768. Note that this includes dummy features at padded gazing positions!

# Only keep the features at non-padded gazing positions
last_hidden_state = [f[~if_pad] for f, if_pad in zip(siglip_outputs.last_hidden_state, gaze_outputs['if_padded_gazing'])]  # list of non-padded features for each video. The reason we use list at batch dimension is because the number of non-padded features might be different for different videos.
```

Something to note in this code:

- Here we used a SigLIP with the same input resolution and patch size as what AutoGaze is trained on. We will introduce below how to use SigLIP with different resolution or patch size.

- In the end, remember to keep only the features at non-padded gazing positions in the end. The features at padded gazing positions are meaningless.

## 🚀 Advanced Usage 🚀

### Applying to Vision Encoders With Any Patch Size or Input Size

AutoGaze predicts patch ids assuming that the video has 224x224 resolution and 16x16 patch size. In the example above, we used SigLIP-Base with 224x224 input resolution and 16x16 patch size, which is exactly compatible. Nevertheless, we can apply AutoGaze to vision encoders with any input resolution and patch size. **All you need to do is to pass the input resolution and patch size through `target_scales` and `target_patch_size` to AutoGaze when calling it**.

Let's take SigLIP2-SO400M with 384x384 input resolution and 14x14 patch size as an example. We first make the input resolution to be 392x392 since 384 is not dividible by 14. To make it multiscale, we use scales of **56x56, 112x112, 196x196, and 392x392**. Note that AutoGaze is trained on scales of 32x32, 64x64, 112x112, and 224x224, and we should always use the same number of scales.

```python
# Load AutoGaze preprocessor with size of 392
autogaze_392_transform = AutoGazeImageProcessor.from_pretrained("nvidia/AutoGaze", size=(392, 392))

# Preprocess the video with AutoGaze preprocessor
video_input_autogaze_392 = transform_video_for_pytorch(raw_video, autogaze_392_transform)  # T * C * H * W
video_input_autogaze_392 = video_input_autogaze_392[None]  # B * T * C * H * W

# Run the AutoGaze model
with torch.inference_mode():
    gaze_outputs_392 = autogaze_model({"video": video_input_autogaze_392}, gazing_ratio=0.75, task_loss_requirement=0.7, target_scales=[56, 112, 196, 392], target_patch_size=14)  # To run AutoGaze for vision encoders with any resolution and patch size, just pass in the target image scales and patch size into AutoGaze.

# Load SigLIP with 384x384 resolution and 14x14 patch size.
siglip_384_transform = AutoImageProcessor.from_pretrained("google/siglip2-so400m-patch14-384")
siglip_384_model = SiglipVisionModel.from_pretrained("google/siglip2-so400m-patch14-384", scales="56+112+196+392", attn_implementation="sdpa")  # Here we make the model to process multiple scales at 56x56, 112x112, 196x196, and 392x392

# Preprocess the video with SigLIP preprocessor
video_input_siglip_384 = transform_video_for_pytorch(raw_video, siglip_384_transform)  # T * C * H * W
video_input_siglip_384 = video_input_siglip_384[None]  # B * T * C * H * W

# Encode the gazed patches with SigLIP
siglip_outputs_384 = siglip_384_model(video_input_siglip_384, gazing_info=gaze_outputs_392)
```


### Running AutoGaze on Any-Resolution, Any-Duration Videos

Even though AutoGaze is only trained on 16-frame 224x224 videos, we can still run it on high-resoltuion, long-form videos. To achieve this, we simply chop the video into 16-frame 224x224 chunks and run on each chunk separately (similar to AnyRes in image MLLMs). Note that this is not limited to 224x224 input resolution. If you want to run it for vision encoders with different resolution or patch size (e.g., SigLIP with 384 resolution and 14x14 patch size, like shown in the example above), you can also chop the video into 16-frame 384x384 chunks.

```python
from einops import rearrange

# A dummy video with 256 frames and 1344x1344 resolution
dummy_video = torch.randn(1, 256, 3, 1344, 1344)

# Chop it into 16-frame, 224x224 chunks
dummy_video_chunks = rearrange(dummy_video, 'B (nt t) C (nh h) (nw w) -> (B nt nh nw) t C h w', t=16, h=224, w=224)

# Run AutoGaze and SigLIP
with torch.inference_mode():
    gaze_outputs_dummy = autogaze_model({"video": dummy_video_chunks}, gazing_ratio=0.75, task_loss_requirement=0.7)
siglip_outputs_dummy = siglip_model(dummy_video_chunks, gazing_info=gaze_outputs_dummy)
```

### Running AutoGaze on Streaming Videos

Since AutoGaze is causal on frame dimension, it can natually process streaming videos. Specifically, it supports passing in one frame at a time along with cache from previous frames, similar to kv cache in LLM.

```python
streaming_gaze_outputs = []
past_inputs_embeds = None
past_attention_mask = None
past_key_values = None
past_conv_values = None
# Treat the video as streaming at loop over its frames
for t in range(video_input_autogaze.shape[1]):
    video_t = video_input_autogaze[:, t:t+1]
    streaming_gaze_outputs_t = autogaze_model(
        {'video': video_t}, 
        gazing_ratio=gazing_ratio, 
        generate_only=True, 
        use_cache=True,
        past_key_values=past_key_values, 
        past_inputs_embeds=past_inputs_embeds,
        past_attention_mask=past_attention_mask, 
        past_conv_values=past_conv_values
    )  # Passing one frame at a time along with cache from history
    streaming_gaze_outputs.append(streaming_gaze_outputs_t)
    past_key_values = streaming_gaze_outputs_t['past_key_values']
    past_inputs_embeds = streaming_gaze_outputs_t['past_input_embeds']
    past_attention_mask = streaming_gaze_outputs_t['past_attention_mask']
    past_conv_values = streaming_gaze_outputs_t['past_conv_values']

# Gather the gazing from all frames
streaming_gazing_pos = [outputs['gazing_pos'] for outputs in streaming_gaze_outputs]

# In streaming case, each gazed patch id is counted from the first patch within the same frame, but in static video case, it counted from the first patch of the whole video. Therefore, to compare with the static video gazing, we need to recalibrate the patch ids.
streaming_gazing_pos = [pos + autogaze_model.num_vision_tokens_each_frame * t for t, pos in enumerate(streaming_gazing_pos)]
streaming_gazing_pos = torch.cat(streaming_gazing_pos, dim=1)

# Check if the gaze output in streaming case is the same as the static case.
assert torch.allclose(gaze_outputs['gazing_pos'], streaming_gazing_pos)
```