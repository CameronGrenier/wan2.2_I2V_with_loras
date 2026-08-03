# Use specific version of nvidia cuda image
FROM wlsdml1114/engui_genai-base_blackwell:1.1 as runtime

# ---------------------------------------------------------------------------
# Pinned revisions.
#
# Every one of these was previously cloned at HEAD, so each rebuild pulled a
# different tree and the image was never reproducible. The SHAs below were read
# off a worker that builds and runs correctly.
#
# ComfyUI is pinned by SHA rather than by tag: the working tree is
# v0.29.0-42-gb53e247c, i.e. 42 commits past the v0.29.0 tag and not equal to
# any release. Checking out a tag would silently change the version.
#
# To move forward later: bump one pin at a time, rebuild, test. Bumping them
# all at once reproduces the original problem.
# ---------------------------------------------------------------------------
ARG COMFYUI_SHA=b53e247c94f9225dc206bcfef5d64a2f7bc85232
ARG GGUF_SHA=6ea2651e7df66d7585f6ffee804b20e92fb38b8a
ARG KJNODES_SHA=4d46ac107c33ed8a3d181b8776ede66498583380
ARG VHS_SHA=4ee72c065db22c9d96c2427954dc69e7b908444b
ARG GGUF_FANTASYTALKING_SHA=48dd427c5cf28cc849467dc97e54515286df0aa0
ARG WANBLOCKSWAP_SHA=5fa2ec0fa55879fe43a33e762fff91fc2c553a67
ARG WANVIDEOWRAPPER_SHA=088128b224242e110d3906c6750e9a3a348a659b
ARG INTELLIGENTVRAM_SHA=3a3fdb41c1b0e01545d9d394304adc846cdde52b
ARG AUTOWAN_SHA=d4f7e6294fc8d1f38c8b3acdb520c64d983099a1
ARG ADAPTIVEWINDOW_SHA=6c46e055f63b031324a0d19f6e2adebcbe76b90b

# Set to 1 to reinstate ComfyUI-Manager. It is off by default: on a headless
# serverless worker running a fixed workflow there is no UI to use it from, and
# it fetches the node registry over the network on every cold start.
ARG INSTALL_COMFYUI_MANAGER=0
ARG COMFYUI_MANAGER_SHA=d404e6234acd609da830ebb9f01e3c975313473e

# Fail the build on any error in a RUN line rather than carrying on with a
# half-populated image.
SHELL ["/bin/bash", "-o", "pipefail", "-e", "-c"]

RUN pip install -U "huggingface_hub[hf_transfer]"
RUN pip install runpod websocket-client

WORKDIR /

RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd /ComfyUI && \
    git checkout ${COMFYUI_SHA} && \
    pip install -r requirements.txt

RUN if [ "${INSTALL_COMFYUI_MANAGER}" = "1" ]; then \
      cd /ComfyUI/custom_nodes && \
      git clone https://github.com/Comfy-Org/ComfyUI-Manager.git && \
      cd ComfyUI-Manager && \
      git checkout ${COMFYUI_MANAGER_SHA} && \
      pip install -r requirements.txt; \
    else \
      echo "Skipping ComfyUI-Manager (INSTALL_COMFYUI_MANAGER=${INSTALL_COMFYUI_MANAGER})"; \
    fi

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/city96/ComfyUI-GGUF && \
    cd ComfyUI-GGUF && \
    git checkout ${GGUF_SHA} && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-KJNodes && \
    cd ComfyUI-KJNodes && \
    git checkout ${KJNODES_SHA} && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite && \
    cd ComfyUI-VideoHelperSuite && \
    git checkout ${VHS_SHA} && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kael558/ComfyUI-GGUF-FantasyTalking && \
    cd ComfyUI-GGUF-FantasyTalking && \
    git checkout ${GGUF_FANTASYTALKING_SHA} && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/orssorbit/ComfyUI-wanBlockswap && \
    cd ComfyUI-wanBlockswap && \
    git checkout ${WANBLOCKSWAP_SHA}

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper && \
    cd ComfyUI-WanVideoWrapper && \
    git checkout ${WANVIDEOWRAPPER_SHA} && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/eddyhhlure1Eddy/IntelligentVRAMNode && \
    cd IntelligentVRAMNode && \
    git checkout ${INTELLIGENTVRAM_SHA}

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/eddyhhlure1Eddy/auto_wan2.2animate_freamtowindow_server && \
    cd auto_wan2.2animate_freamtowindow_server && \
    git checkout ${AUTOWAN_SHA}

# This repo nests its package one directory deeper than ComfyUI expects, so the
# contents are lifted up a level after checkout.
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/eddyhhlure1Eddy/ComfyUI-AdaptiveWindowSize && \
    cd ComfyUI-AdaptiveWindowSize && \
    git checkout ${ADAPTIVEWINDOW_SHA} && \
    cd ComfyUI-AdaptiveWindowSize && \
    mv * ../

# ---------------------------------------------------------------------------
# Model downloads.
#
# `-q` is dropped and each download is size-checked. Previously a failed or
# truncated fetch left a missing or partial file, the build still reported
# success, and the failure only surfaced at prompt-validation time as an opaque
# HTTP 400 from ComfyUI.
#
# These layers sit above `COPY . .` on purpose: editing handler.py or a
# workflow JSON reuses them from cache instead of re-downloading ~40 GB.
# ---------------------------------------------------------------------------
RUN wget --progress=dot:giga \
      https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/I2V/Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors \
      -O /ComfyUI/models/diffusion_models/Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors && \
    test -s /ComfyUI/models/diffusion_models/Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors

RUN wget --progress=dot:giga \
      https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/I2V/Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors \
      -O /ComfyUI/models/diffusion_models/Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors && \
    test -s /ComfyUI/models/diffusion_models/Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors

RUN wget --progress=dot:giga \
      https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors \
      -O /ComfyUI/models/loras/high_noise_model.safetensors && \
    test -s /ComfyUI/models/loras/high_noise_model.safetensors

RUN wget --progress=dot:giga \
      https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors \
      -O /ComfyUI/models/loras/low_noise_model.safetensors && \
    test -s /ComfyUI/models/loras/low_noise_model.safetensors

RUN wget --progress=dot:giga \
      https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors \
      -O /ComfyUI/models/clip_vision/clip_vision_h.safetensors && \
    test -s /ComfyUI/models/clip_vision/clip_vision_h.safetensors

RUN wget --progress=dot:giga \
      https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors \
      -O /ComfyUI/models/text_encoders/umt5-xxl-enc-bf16.safetensors && \
    test -s /ComfyUI/models/text_encoders/umt5-xxl-enc-bf16.safetensors

RUN wget --progress=dot:giga \
      https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors \
      -O /ComfyUI/models/vae/Wan2_1_VAE_bf16.safetensors && \
    test -s /ComfyUI/models/vae/Wan2_1_VAE_bf16.safetensors

# LoadImage resolves names against this directory, and the handler stages every
# input image into it. Created here so the first request never races on mkdir.
RUN mkdir -p /ComfyUI/input

# Report what actually landed in the image, so a bad build is visible in the
# build log rather than at request time.
RUN echo "=== pinned revisions ===" && \
    echo "ComfyUI $(git -C /ComfyUI rev-parse HEAD) ($(git -C /ComfyUI describe --tags))" && \
    for d in /ComfyUI/custom_nodes/*/; do \
      [ -d "$d/.git" ] && echo "$(basename $d) $(git -C $d rev-parse HEAD)"; \
    done; \
    echo "=== models ===" && \
    find /ComfyUI/models -name '*.safetensors' -printf '%s\t%p\n' | sort -rn

COPY . .
COPY extra_model_paths.yaml /ComfyUI/extra_model_paths.yaml
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]