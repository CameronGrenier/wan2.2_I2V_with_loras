import runpod
from runpod.serverless.utils import rp_upload
import os
import websocket
import base64
import json
import uuid
import logging
import random
import urllib.request
import urllib.error
import urllib.parse
import binascii  # imported for Base64 error handling
import shutil
import subprocess
import time

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server_address = os.getenv('SERVER_ADDRESS', '127.0.0.1')
client_id = str(uuid.uuid4())

# ---------------------------------------------------------------------------
# Node map for new_Wan22_api.json / new_Wan22_flf2v_api.json
#
#   244 LoadImage                    -> image
#   617 LoadImage                    -> end image (FLF2V workflow only)
#   135 WanVideoTextEncode           -> positive_prompt / negative_prompt
#   235 INTConstant                  -> width
#   236 INTConstant                  -> height
#   541 WanVideoImageToVideoEncode   -> num_frames
#   498 WanVideoContextOptions       -> context_frames / context_overlap
#   220 WanVideoSampler (HIGH)       -> seed;  steps <- 569, end_step   <- 575
#   540 WanVideoSampler (LOW)        -> seed;  steps <- 569, start_step <- 575
#   569 INTConstant                  -> total step count
#   575 INTConstant                  -> high/low split point
#   570 CreateCFGScheduleFloatList   -> CFG schedule feeding node 220
#   279 WanVideoLoraSelectMulti      -> HIGH LoRA slots (lora_0 is Lightning)
#   553 WanVideoLoraSelectMulti      -> LOW  LoRA slots (lora_0 is Lightning)
#
# Two corrections against the upstream handler:
#
#  1. CFG was written to node 540, the LOW-noise sampler. That sampler is
#     Lightning-distilled and ships at cfg 1.0 on purpose. Any value above 1.0
#     doubles the forward passes per step (classifier-free guidance runs the
#     model twice) for guidance the distilled pass was not trained to use, so
#     it cost roughly 2x on half the run AND degraded output. CFG belongs on
#     node 570, whose schedule feeds the HIGH-noise sampler.
#
#  2. The steps block targeted nodes 834 / 829, which belong to the ksampler
#     branch's workflow and do not exist here. Guarded by `if "834" in prompt`,
#     so it silently did nothing. The real controls are nodes 569 and 575.
# ---------------------------------------------------------------------------
NODE_IMAGE = "244"
NODE_END_IMAGE = "617"
NODE_TEXT = "135"
NODE_WIDTH = "235"
NODE_HEIGHT = "236"
NODE_I2V_ENCODE = "541"
NODE_CONTEXT = "498"
NODE_SAMPLER_HIGH = "220"
NODE_SAMPLER_LOW = "540"
NODE_STEPS = "569"
NODE_SPLIT = "575"
NODE_CFG_SCHEDULE = "570"
NODE_LORA_HIGH = "279"
NODE_LORA_LOW = "553"

MAX_SEED = 2**53 - 1

# ComfyUI resolves LoadImage names against this directory.
COMFY_INPUT_DIR = os.getenv("COMFY_INPUT_DIR", "/ComfyUI/input")

# Fraction of total steps handled by the high-noise expert. The shipped graph
# uses 4 of 8. Composition and motion are settled in this half; detail and
# refinement happen in the low-noise half.
HIGH_NOISE_FRACTION = 0.5

DEFAULT_NEGATIVE = (
    "bright tones, overexposed, static, blurred details, subtitles, style, "
    "works, paintings, images, static, overall gray, worst quality, low quality, "
    "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)


def to_nearest_multiple_of_16(value):
    """Round the given value to the nearest multiple of 16, minimum 16."""
    try:
        numeric_value = float(value)
    except Exception:
        raise Exception(f"width/height value is not a number: {value}")
    adjusted = int(round(numeric_value / 16.0) * 16)
    if adjusted < 16:
        adjusted = 16
    return adjusted


def process_input(input_data, temp_dir, output_filename, input_type):
    """Process the input data and return a usable file path."""
    if input_type == "path":
        logger.info(f"Handling path input: {input_data}")
        return input_data
    elif input_type == "url":
        logger.info(f"Handling URL input: {input_data}")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        return download_file_from_url(input_data, file_path)
    elif input_type == "base64":
        logger.info("Handling Base64 input")
        return save_base64_to_file(input_data, temp_dir, output_filename)
    else:
        raise Exception(f"Unsupported input type: {input_type}")


def download_file_from_url(url, output_path):
    """Download a file from a URL."""
    try:
        result = subprocess.run(
            ['wget', '-O', output_path, '--no-verbose', url],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            logger.info(f"Downloaded file from URL: {url} -> {output_path}")
            return output_path
        else:
            logger.error(f"wget download failed: {result.stderr}")
            raise Exception(f"URL download failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.error("Download timed out")
        raise Exception("Download timed out")
    except Exception as e:
        logger.error(f"Error during download: {e}")
        raise Exception(f"Error during download: {e}")


def save_base64_to_file(base64_data, temp_dir, output_filename):
    """Decode Base64 data and save it to a file."""
    try:
        # Strip the data URI prefix if present. The README's own examples send
        # "data:image/jpeg;base64,...", which the original code fed straight to
        # b64decode and crashed on.
        if isinstance(base64_data, str) and base64_data.startswith("data:"):
            base64_data = base64_data.split(",", 1)[-1]

        decoded_data = base64.b64decode(base64_data)
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        with open(file_path, 'wb') as f:
            f.write(decoded_data)
        logger.info(f"Saved Base64 input to file: '{file_path}'")
        return file_path
    except (binascii.Error, ValueError) as e:
        logger.error(f"Base64 decode failed: {e}")
        raise Exception(f"Base64 decode failed: {e}")


def stage_for_comfy(src_path, dest_name):
    """Copy an image into ComfyUI's input directory, return the bare filename.

    LoadImage resolves and validates its `image` field against ComfyUI's own
    input directory. Current ComfyUI rejects absolute paths that point outside
    it, which is why passing /task_<uuid>/input_image.jpg produced:

        LoadImage 244: Invalid image file: /task_.../input_image.jpg

    Anything that arrives as base64, a URL download, or a network-volume path
    has to be copied here first and referenced by name.
    """
    if not os.path.exists(src_path):
        raise Exception(f"Input image does not exist: {src_path}")

    size = os.path.getsize(src_path)
    if size == 0:
        raise Exception(f"Input image is empty (0 bytes): {src_path}")

    os.makedirs(COMFY_INPUT_DIR, exist_ok=True)
    dest_path = os.path.join(COMFY_INPUT_DIR, dest_name)
    shutil.copyfile(src_path, dest_path)
    logger.info(f"Staged input image: {src_path} -> {dest_path} ({size} bytes)")
    return dest_name


# ---------------------------------------------------------------------------
# REPLACE the existing handler() with this one.
# Requires `import shutil` at the top of the file.
# ---------------------------------------------------------------------------


def queue_prompt(prompt):
    url = f"http://{server_address}:8188/prompt"
    logger.info(f"Queueing prompt to: {url}")
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(url, data=data)
    try:
        return json.loads(urllib.request.urlopen(req).read())
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode('utf-8', 'replace')
        except Exception:
            body = "<no response body>"
        logger.error(f"ComfyUI rejected the prompt ({e.code}): {body}")
        raise Exception(f"ComfyUI rejected the prompt ({e.code}): {body[:2000]}") from e


# ---------------------------------------------------------------------------
# NEW helper. Put it next to process_input.
# ---------------------------------------------------------------------------

def get_image(filename, subfolder, folder_type):
    url = f"http://{server_address}:8188/view"
    logger.info(f"Getting image from: {url}")
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"{url}?{url_values}") as response:
        return response.read()


def get_history(prompt_id):
    url = f"http://{server_address}:8188/history/{prompt_id}"
    logger.info(f"Getting history from: {url}")
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())


def get_videos(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_videos = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break
        else:
            continue

    history = get_history(prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        videos_output = []
        if 'gifs' in node_output:
            for video in node_output['gifs']:
                with open(video['fullpath'], 'rb') as f:
                    video_data = base64.b64encode(f.read()).decode('utf-8')
                videos_output.append(video_data)
        output_videos[node_id] = videos_output

    return output_videos


def load_workflow(workflow_path):
    with open(workflow_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def resolve_seed(job_input):
    """Return the seed to use. Omitted or -1 means a fresh random one."""
    raw_seed = job_input.get("seed", None)
    if raw_seed is None or raw_seed == -1:
        seed = random.randint(0, MAX_SEED)
        logger.info(f"No seed supplied -> using random seed: {seed}")
        return seed
    try:
        seed = int(raw_seed) % (MAX_SEED + 1)
    except Exception:
        raise Exception(f"seed value is not an integer: {raw_seed}")
    logger.info(f"Seed applied: {seed}")
    return seed


def apply_steps(prompt, steps):
    """Set the total step count and the high/low split point.

    Node 569 feeds `steps` on both samplers and `steps` on the CFG schedule.
    Node 575 feeds `end_step` on the high sampler and `start_step` on the low
    sampler, so it is the boundary between the two experts.
    """
    try:
        total_steps = int(steps)
    except Exception:
        raise Exception(f"steps value is not an integer: {steps}")
    if total_steps < 2:
        logger.warning(f"steps={total_steps} is too small. Clamping to 2.")
        total_steps = 2

    split_at = max(1, min(total_steps - 1, round(total_steps * HIGH_NOISE_FRACTION)))

    if NODE_STEPS in prompt:
        prompt[NODE_STEPS]["inputs"]["value"] = total_steps
    else:
        logger.warning(f"Node {NODE_STEPS} not found, total steps not applied.")
    if NODE_SPLIT in prompt:
        prompt[NODE_SPLIT]["inputs"]["value"] = split_at
    else:
        logger.warning(f"Node {NODE_SPLIT} not found, split point not applied.")

    logger.info(
        f"Steps applied: {total_steps} total "
        f"(high-noise 0-{split_at} / low-noise {split_at}-end)"
    )
    return total_steps


def apply_cfg(prompt, cfg):
    """Set the CFG schedule that feeds the HIGH-noise sampler.

    The LOW-noise sampler stays at its shipped 1.0. Raising it there does not
    improve prompt adherence, it just runs the distilled model twice per step.
    """
    try:
        cfg_value = float(cfg)
    except Exception:
        raise Exception(f"cfg value is not a number: {cfg}")

    if NODE_CFG_SCHEDULE in prompt:
        prompt[NODE_CFG_SCHEDULE]["inputs"]["cfg_scale_start"] = cfg_value
        prompt[NODE_CFG_SCHEDULE]["inputs"]["cfg_scale_end"] = cfg_value
        logger.info(f"CFG applied to high-noise schedule (node {NODE_CFG_SCHEDULE}): {cfg_value}")
    else:
        logger.warning(
            f"Node {NODE_CFG_SCHEDULE} (CreateCFGScheduleFloatList) not found, "
            f"CFG not applied."
        )
    return cfg_value


def apply_loras(prompt, lora_pairs):
    """Write user LoRAs into the HIGH (279) and LOW (553) selector nodes.

    Slot 0 on each node holds the Lightning distillation LoRA and must not be
    touched, so user LoRAs start at slot 1. Each node has slots 0-4, leaving
    four usable slots.
    """
    if not lora_pairs:
        return 0

    if NODE_LORA_HIGH not in prompt or NODE_LORA_LOW not in prompt:
        logger.warning("LoRA selector nodes not found, LoRAs not applied.")
        return 0

    applied = 0
    for i, lora_pair in enumerate(lora_pairs[:4]):
        slot = i + 1  # slot 0 is reserved for the Lightning LoRA
        lora_high = lora_pair.get("high")
        lora_low = lora_pair.get("low")
        high_weight = lora_pair.get("high_weight", 1.0)
        low_weight = lora_pair.get("low_weight", 1.0)

        if lora_high:
            prompt[NODE_LORA_HIGH]["inputs"][f"lora_{slot}"] = lora_high
            prompt[NODE_LORA_HIGH]["inputs"][f"strength_{slot}"] = high_weight
            logger.info(
                f"LoRA {slot} HIGH -> node {NODE_LORA_HIGH}: {lora_high} @ {high_weight}"
            )
            applied += 1

        if lora_low:
            prompt[NODE_LORA_LOW]["inputs"][f"lora_{slot}"] = lora_low
            prompt[NODE_LORA_LOW]["inputs"][f"strength_{slot}"] = low_weight
            logger.info(
                f"LoRA {slot} LOW  -> node {NODE_LORA_LOW}: {lora_low} @ {low_weight}"
            )

    return applied


def handler(job):
    job_input = job.get("input", {})

    logger.info(f"Received job input: { {k: v for k, v in job_input.items() if 'base64' not in k} }")
    task_id = f"task_{uuid.uuid4()}"

    # ---- image input (use only one of path / url / base64) ----------------
    if "image_path" in job_input:
        image_path = process_input(job_input["image_path"], task_id, "input_image.jpg", "path")
    elif "image_url" in job_input:
        image_path = process_input(job_input["image_url"], task_id, "input_image.jpg", "url")
    elif "image_base64" in job_input:
        image_path = process_input(job_input["image_base64"], task_id, "input_image.jpg", "base64")
    else:
        image_path = "/example_image.png"
        logger.info("Using the default image file: /example_image.png")

    # LoadImage only accepts names inside ComfyUI's input directory.
    image_name = stage_for_comfy(image_path, f"{task_id}_input.jpg")

    # ---- end image input (FLF2V) -----------------------------------------
    end_image_path_local = None
    if "end_image_path" in job_input:
        end_image_path_local = process_input(job_input["end_image_path"], task_id, "end_image.jpg", "path")
    elif "end_image_url" in job_input:
        end_image_path_local = process_input(job_input["end_image_url"], task_id, "end_image.jpg", "url")
    elif "end_image_base64" in job_input:
        end_image_path_local = process_input(job_input["end_image_base64"], task_id, "end_image.jpg", "base64")

    end_image_name = None
    if end_image_path_local:
        end_image_name = stage_for_comfy(end_image_path_local, f"{task_id}_end.jpg")

    # ---- workflow selection ----------------------------------------------
    lora_pairs = job_input.get("lora_pairs", []) or []
    if len(lora_pairs) > 4:
        logger.warning(
            f"{len(lora_pairs)} LoRA pairs supplied. Only 4 are supported. Using the first 4."
        )
        lora_pairs = lora_pairs[:4]

    workflow_file = "/new_Wan22_flf2v_api.json" if end_image_name else "/new_Wan22_api.json"
    logger.info(
        f"Using {'FLF2V' if end_image_name else 'single'} workflow "
        f"with {len(lora_pairs)} LoRA pair(s)"
    )
    prompt = load_workflow(workflow_file)

    # ---- parameters -------------------------------------------------------
    # All read with .get() and a default. The original indexed job_input
    # directly for prompt/seed/cfg/width/height, so omitting any one of them
    # raised a KeyError and failed the job.
    length = job_input.get("length", 81)
    steps = job_input.get("steps", 8)
    cfg = job_input.get("cfg", 2.0)
    seed = resolve_seed(job_input)

    positive_prompt = job_input.get("prompt", "")
    if not positive_prompt:
        raise Exception("A 'prompt' is required.")
    negative_prompt = job_input.get("negative_prompt", DEFAULT_NEGATIVE)

    original_width = job_input.get("width", 480)
    original_height = job_input.get("height", 832)
    adjusted_width = to_nearest_multiple_of_16(original_width)
    adjusted_height = to_nearest_multiple_of_16(original_height)
    if adjusted_width != original_width:
        logger.info(f"Width adjusted to nearest multiple of 16: {original_width} -> {adjusted_width}")
    if adjusted_height != original_height:
        logger.info(f"Height adjusted to nearest multiple of 16: {original_height} -> {adjusted_height}")

    # ---- write into the graph --------------------------------------------
    # Bare filename, not an absolute path: LoadImage resolves it against
    # ComfyUI's input directory.
    prompt[NODE_IMAGE]["inputs"]["image"] = image_name
    prompt[NODE_TEXT]["inputs"]["positive_prompt"] = positive_prompt
    prompt[NODE_TEXT]["inputs"]["negative_prompt"] = negative_prompt
    prompt[NODE_WIDTH]["inputs"]["value"] = adjusted_width
    prompt[NODE_HEIGHT]["inputs"]["value"] = adjusted_height
    prompt[NODE_I2V_ENCODE]["inputs"]["num_frames"] = length
    prompt[NODE_CONTEXT]["inputs"]["context_frames"] = length
    prompt[NODE_CONTEXT]["inputs"]["context_overlap"] = job_input.get("context_overlap", 48)
    prompt[NODE_SAMPLER_HIGH]["inputs"]["seed"] = seed
    prompt[NODE_SAMPLER_LOW]["inputs"]["seed"] = seed

    applied_steps = apply_steps(prompt, steps)
    applied_cfg = apply_cfg(prompt, cfg)

    if end_image_name:
        if NODE_END_IMAGE in prompt:
            prompt[NODE_END_IMAGE]["inputs"]["image"] = end_image_name
        else:
            logger.warning(
                f"Node {NODE_END_IMAGE} not found in {workflow_file}, end image not applied."
            )

    apply_loras(prompt, lora_pairs)

    # ---- run --------------------------------------------------------------
    ws_url = f"ws://{server_address}:8188/ws?clientId={client_id}"
    logger.info(f"Connecting to WebSocket: {ws_url}")

    http_url = f"http://{server_address}:8188/"
    logger.info(f"Checking HTTP connection to: {http_url}")

    max_http_attempts = 180
    for http_attempt in range(max_http_attempts):
        try:
            response = urllib.request.urlopen(http_url, timeout=5)
            logger.info(f"HTTP connection succeeded (attempt {http_attempt+1})")
            break
        except Exception as e:
            logger.warning(f"HTTP connection failed (attempt {http_attempt+1}/{max_http_attempts}): {e}")
            if http_attempt == max_http_attempts - 1:
                raise Exception("Could not connect to the ComfyUI server. Check that it is running.")
            time.sleep(1)

    ws = websocket.WebSocket()
    max_attempts = int(180 / 5)
    for attempt in range(max_attempts):
        try:
            ws.connect(ws_url)
            logger.info(f"WebSocket connection succeeded (attempt {attempt+1})")
            break
        except Exception as e:
            logger.warning(f"WebSocket connection failed (attempt {attempt+1}/{max_attempts}): {e}")
            if attempt == max_attempts - 1:
                raise Exception("WebSocket connection timed out (3 minutes)")
            time.sleep(5)

    try:
        videos = get_videos(ws, prompt)
    finally:
        ws.close()
        # Staged copies are per-job and never reused. Left in place they
        # accumulate in the image layer for the life of the worker.
        for name in (image_name, end_image_name):
            if not name:
                continue
            try:
                os.remove(os.path.join(COMFY_INPUT_DIR, name))
            except OSError:
                pass

    for node_id in videos:
        if videos[node_id]:
            # Echo back the settings actually used so a good result can be
            # reproduced. Extra keys are ignored by clients that only read
            # "video".
            return {
                "video": videos[node_id][0],
                "seed": seed,
                "steps": applied_steps,
                "cfg": applied_cfg,
                "width": adjusted_width,
                "height": adjusted_height,
                "length": length,
            }

    return {"error": "Video not found."}


runpod.serverless.start({"handler": handler})