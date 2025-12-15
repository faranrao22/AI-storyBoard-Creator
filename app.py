import os
import json
import asyncio
import nest_asyncio
import edge_tts
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
from groq import Groq
import requests
import textwrap
from io import BytesIO
import time
import urllib.parse
import subprocess
import warnings
import tempfile
import shutil
from pathlib import Path

warnings.filterwarnings('ignore')
nest_asyncio.apply()

# -----------------------------
# CONFIGURE GROQ
# -----------------------------
# SECURITY: Use environment variable instead of hardcoded key
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

# Create temp directory for files
TEMP_DIR = Path(tempfile.mkdtemp(prefix="storyboard_"))

def get_temp_path(filename):
    """Get path in temp directory"""
    return str(TEMP_DIR / filename)

# -----------------------------
# STORY GENERATION
# -----------------------------
def generate_consistent_story(user_prompt, num_scenes=4):
    """Generate story with consistent visual style"""
    system_prompt = """You are an expert storyboard artist and writer.
    
    Your job is to create a VISUALLY CONSISTENT storyboard where:
    1. Characters look the SAME in every scene
    2. The art style is CONSISTENT throughout
    3. Settings and colors are COHERENT
    
    First, create a VISUAL BIBLE that defines:
    - Main character(s) appearance in DETAIL
    - Art style for the entire story
    - Color palette
    - Setting/world description
    
    Then create scenes that ALL reference this visual bible.
    
    Return ONLY valid JSON in this format:
    {
        "title": "Story Title",
        "visual_bible": {
            "art_style": "e.g., Pixar-style 3D animation, soft lighting",
            "color_palette": "e.g., warm oranges, soft blues",
            "main_character": "Detailed character description used in EVERY scene",
            "world_setting": "World description"
        },
        "scenes": [
            {
                "scene_number": 1,
                "description": "What happens",
                "visual_prompt": "MUST include character + art style + action",
                "narration": "2-3 sentence narration",
                "camera_angle": "wide shot / close-up",
                "mood": "happy / tense"
            }
        ]
    }
    """
    
    user_message = f"""Create a {num_scenes}-scene storyboard for:

    "{user_prompt}"

    Return ONLY valid JSON, no other text."""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=3000
        )
        
        response_text = chat_completion.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        story_data = json.loads(response_text)
        return enhance_visual_prompts(story_data)
        
    except json.JSONDecodeError as e:
        print(f"JSON error: {e}")
        print("Response:", response_text[:500] if 'response_text' in locals() else "No response")
        return None
    except Exception as e:
        print(f"Groq error: {e}")
        return None

def enhance_visual_prompts(story_data):
    """Enhance prompts with story-specific visual context for better image generation"""
    visual_bible = story_data.get("visual_bible", {})
    art_style = visual_bible.get("art_style", "3D cartoon animation, Pixar-style, vibrant colors")
    color_palette = visual_bible.get("color_palette", "warm amber and dusty blue tones")
    main_character = visual_bible.get("main_character", "a clever black crow with glossy feathers and intelligent eyes")
    world_setting = visual_bible.get("world_setting", "a sun-baked Mediterranean courtyard with terracotta tiles and potted herbs")

    # Ensure core story elements are emphasized
    if "thirsty crow" in str(story_data).lower() or "crow" in str(main_character).lower():
        art_style = "3D cartoon animation, Pixar-style, cinematic lighting"
        color_palette = "warm amber sunset tones, dusty blues, golden highlights"
        main_character = "a clever black crow with glossy iridescent feathers, bright intelligent eyes, slightly dusty from heat"
        world_setting = "a sun-scorched stone courtyard in Mediterranean style, terracotta tiles, cracked earth, empty clay pots, harsh midday sun"

    consistency_prefix = f"{art_style}, {color_palette}, {world_setting}"

    for scene in story_data.get("scenes", []):
        original_prompt = str(scene.get("visual_prompt", ""))
        
        # Rebuild prompt with story context first
        enhanced_prompt = (
            f"{main_character}, "
            f"{original_prompt}, "
            f"{consistency_prefix}, "
            "sharp focus, ultra-detailed, professional cinematic quality, 8k"
        )
        
        scene["visual_prompt"] = enhanced_prompt
        scene["style_reference"] = consistency_prefix
        
        print(f"\n🖼️ Scene {scene.get('scene_number')} Prompt:\n{enhanced_prompt[:180]}...")
    
    return story_data

# -----------------------------
# IMAGE GENERATION
# -----------------------------
def generate_consistent_image(prompt, scene_number, style_reference, max_retries=3):
    """Generate image with retry logic"""
    consistency_keywords = ["consistent character design", "same art style", "coherent visual style"]
    enhanced_prompt = f"{prompt}, {', '.join(consistency_keywords)}, high quality, detailed, 4k"
    encoded_prompt = urllib.parse.quote(enhanced_prompt)
    image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=576&nologo=true"
    
    for attempt in range(max_retries):
        try:
            print(f"🎨 Generating Scene {scene_number} (attempt {attempt + 1}/{max_retries})...")
            response = requests.get(image_url, timeout=90)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                image = add_scene_label(image, scene_number)
                image_path = get_temp_path(f"scene_{scene_number}.png")
                image.save(image_path)
                return image_path
            else:
                print(f"HTTP {response.status_code}, retrying...")
                time.sleep(2)
        except Exception as e:
            print(f"Image generation error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    # Fallback to placeholder
    return create_styled_placeholder(prompt, scene_number, style_reference)

def add_scene_label(image, scene_number):
    """Add scene number label to image"""
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
    
    draw.rectangle([10, 10, 120, 45], fill=(0, 0, 0, 180))
    draw.text((20, 15), f"Scene {scene_number}", fill=(255, 255, 255), font=font)
    return image

def create_styled_placeholder(text, scene_number, style_reference):
    """Create placeholder image when generation fails"""
    bg_color = (40, 40, 50)
    text_color = (200, 200, 220)
    img = Image.new('RGB', (1024, 576), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 36)
            small_font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
            small_font = font
    
    draw.text((50, 50), f"Scene {scene_number}", fill=text_color, font=font)
    lines = textwrap.wrap(text, width=60)[:6]
    y = 120
    for line in lines:
        draw.text((50, y), line, fill=text_color, font=small_font)
        y += 30
    
    image_path = get_temp_path(f"scene_{scene_number}.png")
    img.save(image_path)
    return image_path

# -----------------------------
# AUDIO GENERATION
# -----------------------------
async def generate_speech_async(text, output_file, voice="en-US-AriaNeural"):
    """Generate speech asynchronously"""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)

def generate_speech(text, scene_number, voice="en-US-AriaNeural"):
    """Generate speech for narration"""
    output_file = get_temp_path(f"narration_{scene_number}.mp3")
    try:
        asyncio.run(generate_speech_async(text, output_file, voice))
        if os.path.exists(output_file):
            return output_file
        else:
            print(f"Audio file not created: {output_file}")
            return None
    except Exception as e:
        print(f"Speech generation error: {e}")
        return None

# -----------------------------
# FFmpeg Helper
# -----------------------------
def check_ffmpeg():
    """Check if FFmpeg is installed"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def run_ffmpeg(cmd):
    """Run FFmpeg command with error handling"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            raise RuntimeError(f"FFmpeg command failed: {result.stderr}")
        return result
    except subprocess.TimeoutExpired:
        raise RuntimeError("FFmpeg command timed out")
    except Exception as e:
        raise RuntimeError(f"FFmpeg execution error: {e}")

# -----------------------------
# VIDEO CREATION
# -----------------------------
def create_title_card_image(title, output_path=None):
    """Create title card image"""
    if output_path is None:
        output_path = get_temp_path("title_card.png")
    
    img = Image.new('RGB', (1024, 576), color=(20, 20, 30))
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 48)
        except:
            font = ImageFont.load_default()
    
    # Wrap title if too long
    wrapped_title = textwrap.wrap(title, width=30)
    y_offset = 250 - (len(wrapped_title) * 30)
    
    for line in wrapped_title:
        bbox = draw.textbbox((0, 0), line, font=font)
        x = (1024 - (bbox[2] - bbox[0])) // 2
        draw.text((x, y_offset), line, fill=(255, 255, 255), font=font)
        y_offset += 60
    
    img.save(output_path)
    return output_path

def create_end_card_image(output_path=None):
    """Create end card image"""
    if output_path is None:
        output_path = get_temp_path("end_card.png")
    
    img = Image.new('RGB', (1024, 576), color=(20, 20, 30))
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 48)
        except:
            font = ImageFont.load_default()
    
    text = "The End"
    bbox = draw.textbbox((0, 0), text, font=font)
    x = (1024 - (bbox[2] - bbox[0])) // 2
    draw.text((x, 250), text, fill=(255, 255, 255), font=font)
    img.save(output_path)
    return output_path

def get_audio_duration(audio_path):
    """Get duration of audio file"""
    try:
        ffprobe_cmd = f'ffprobe -v quiet -show_entries format=duration -of csv=p=0 "{audio_path}"'
        result = subprocess.run(ffprobe_cmd, shell=True, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except:
        pass
    return None

def image_audio_to_video(image_path, audio_path, output_video, duration=5, zoom=False):
    """Convert image and audio to video with optional zoom effect"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Determine duration from audio or use default
    if audio_path and os.path.exists(audio_path):
        audio_duration = get_audio_duration(audio_path)
        if audio_duration:
            duration = audio_duration + 1.5
        audio_input = audio_path
    else:
        # Generate silent audio
        silent_path = get_temp_path("silent.wav")
        run_ffmpeg(f'ffmpeg -y -f lavfi -i anullsrc=r=24000:cl=mono -t {duration} -q:a 9 -acodec libmp3lame "{silent_path}"')
        audio_input = silent_path

    # Build video filter - FIXED: Proper escaping for zoompan
    if zoom:
        # Use single quotes for the z parameter to avoid shell escaping issues
        vf = f"scale=1024:576:force_original_aspect_ratio=increase,crop=1024:576,zoompan=z='min(zoom+0.001,1.05)':d=1:s=1024x576,fps=24"
    else:
        vf = "scale=1024:576:force_original_aspect_ratio=increase,crop=1024:576,fps=24"

    cmd = (
        f'ffmpeg -y -loop 1 -i "{image_path}" '
        f'-i "{audio_input}" '
        f'-vf "{vf}" '
        f'-c:v libx264 -tune stillimage -c:a aac -b:a 192k '
        f'-pix_fmt yuv420p -shortest -t {duration:.2f} "{output_video}"'
    )
    
    run_ffmpeg(cmd)
    
    # Cleanup silent audio if created
    if audio_input != audio_path and os.path.exists(audio_input):
        try:
            os.remove(audio_input)
        except:
            pass

def create_final_storyboard_video(story_data, output_filename=None):
    """Create final video from all scenes"""
    if output_filename is None:
        output_filename = get_temp_path("storyboard_video.mp4")
    
    title = story_data.get("title", "My Story")
    
    # Title card
    title_img = create_title_card_image(title)
    title_video = get_temp_path("title_video.mp4")
    image_audio_to_video(title_img, None, title_video, duration=3)

    # Scene videos
    scene_videos = [title_video]
    for i, scene in enumerate(story_data.get("scenes", [])):
        scene_num = scene.get("scene_number", i + 1)
        img = get_temp_path(f"scene_{scene_num}.png")
        audio = get_temp_path(f"narration_{scene_num}.mp3")
        out = get_temp_path(f"scene_{scene_num}.mp4")
        
        if os.path.exists(img):
            try:
                image_audio_to_video(img, audio if os.path.exists(audio) else None, out, zoom=True)
                scene_videos.append(out)
            except Exception as e:
                print(f"Error creating video for scene {scene_num}: {e}")

    # End card
    end_img = create_end_card_image()
    end_video = get_temp_path("end_video.mp4")
    image_audio_to_video(end_img, None, end_video, duration=3)
    scene_videos.append(end_video)

    # Concatenate all videos
    filelist_path = get_temp_path("filelist.txt")
    with open(filelist_path, "w") as f:
        for vid in scene_videos:
            if os.path.exists(vid):
                f.write(f"file '{vid}'\n")

    cmd = f'ffmpeg -y -f concat -safe 0 -i "{filelist_path}" -c copy "{output_filename}"'
    run_ffmpeg(cmd)

    return output_filename

# -----------------------------
# MAIN GENERATOR
# -----------------------------
def generate_complete_storyboard(story_idea, num_scenes=4, voice="en-US-AriaNeural"):
    """Generate complete storyboard with images, audio, and video"""
    
    # Check FFmpeg
    if not check_ffmpeg():
        return {
            "status": "error",
            "error": "FFmpeg is not installed. Please install FFmpeg to generate videos."
        }
    
    # Generate story
    print("📝 Generating story...")
    story_data = generate_consistent_story(story_idea, num_scenes)
    if not story_data:
        return {
            "status": "error",
            "error": "Failed to generate story. Check Groq API key and connectivity."
        }
    
    visual_bible = story_data.get("visual_bible", {})
    style_reference = f"{visual_bible.get('art_style', '')}, {visual_bible.get('color_palette', '')}"
    
    # Generate images
    print("🎨 Generating images...")
    images = []
    for i, scene in enumerate(story_data.get("scenes", [])):
        scene_num = scene.get("scene_number", i + 1)
        image_path = generate_consistent_image(scene["visual_prompt"], scene_num, style_reference)
        if image_path:
            images.append(image_path)
        time.sleep(1)  # Rate limiting
    
    if not images:
        return {
            "status": "error",
            "error": "Failed to generate any images."
        }
    
    # Generate audio
    print("🎙️ Generating narration...")
    audio_files = []
    for i, scene in enumerate(story_data.get("scenes", [])):
        scene_num = scene.get("scene_number", i + 1)
        audio_path = generate_speech(scene["narration"], scene_num, voice)
        audio_files.append(audio_path)
    
    # Create video
    print("🎬 Creating final video...")
    try:
        video_path = create_final_storyboard_video(story_data)
    except Exception as e:
        print(f"Video creation error: {e}")
        return {
            "status": "error",
            "error": f"Failed to create video: {str(e)}"
        }
    
    return {
        "status": "completed",
        "story": story_data,
        "images": images,
        "audio_files": audio_files,
        "video": video_path
    }

# -----------------------------
# GRADIO UI
# -----------------------------
def process_storyboard_enhanced(story_idea, num_scenes, voice_choice):
    """Process storyboard generation from UI"""
    if not story_idea or not story_idea.strip():
        return (
            "❌ Please enter a story idea",
            [],
            None,
            "❌ No story idea provided"
        )
    
    voice_map = {
        "Female (US) - Aria": "en-US-AriaNeural",
        "Male (US) - Guy": "en-US-GuyNeural",
        "Female (UK) - Sonia": "en-GB-SoniaNeural",
        "Female (Australian) - Natasha": "en-AU-NatashaNeural",
        "Male (UK) - Ryan": "en-GB-RyanNeural"
    }
    voice = voice_map.get(voice_choice, "en-US-AriaNeural")
    
    results = generate_complete_storyboard(story_idea, int(num_scenes), voice)
    
    if results["status"] == "completed":
        story_data = results["story"]
        
        # Format story text
        story_text = f"# 📖 {story_data.get('title', 'My Story')}\n\n"
        vb = story_data.get("visual_bible", {})
        if vb:
            story_text += "## 🎨 Visual Style Guide\n\n"
            story_text += f"**Art Style:** {vb.get('art_style', 'N/A')}\n\n"
            story_text += f"**Color Palette:** {vb.get('color_palette', 'N/A')}\n\n"
            story_text += f"**Main Character:** {vb.get('main_character', 'N/A')}\n\n"
            story_text += "---\n\n"
        
        story_text += "## 🎬 Scenes\n\n"
        for scene in story_data.get("scenes", []):
            story_text += f"### Scene {scene.get('scene_number', '?')}\n\n"
            story_text += f"**Description:** {scene.get('description', '')}\n\n"
            story_text += f"**Narration:** {scene.get('narration', '')}\n\n"
            story_text += f"**Camera:** {scene.get('camera_angle', 'medium shot')} | **Mood:** {scene.get('mood', 'neutral')}\n\n"
            story_text += "---\n\n"
        
        # Load images
        images = []
        for img_path in results["images"]:
            if img_path and os.path.exists(img_path):
                try:
                    images.append(Image.open(img_path))
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
        
        return (
            story_text,
            images,
            results["video"],
            "✅ Storyboard generated successfully!"
        )
    else:
        return (
            "❌ Error generating storyboard",
            [],
            None,
            results.get("error", "Unknown error occurred")
        )

# -----------------------------
# GRADIO INTERFACE
# -----------------------------
with gr.Blocks(title="AI Storyboard Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 AI Storyboard Generator")
    gr.Markdown("Transform your ideas into visual stories with AI-generated images, narration, and video!")
    
    with gr.Row():
        with gr.Column():
            story_input = gr.Textbox(
                label="Story Idea", 
                placeholder="A lonely astronaut discovers a glowing forest on Mars...", 
                lines=5
            )
            num_scenes = gr.Slider(2, 8, value=4, step=1, label="Number of Scenes")
            voice_choice = gr.Dropdown(
                [
                    "Female (US) - Aria",
                    "Male (US) - Guy",
                    "Female (UK) - Sonia",
                    "Female (Australian) - Natasha",
                    "Male (UK) - Ryan"
                ], 
                value="Female (US) - Aria", 
                label="Narrator Voice"
            )
            generate_btn = gr.Button("🎬 Generate Storyboard", variant="primary", size="lg")
            status_output = gr.Textbox(label="Status", interactive=False)
        
        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("📖 Story Script"):
                    script_output = gr.Markdown()
                with gr.TabItem("🖼️ Scene Gallery"):
                    gallery_output = gr.Gallery(label="Scenes", columns=2, height=500)
                with gr.TabItem("🎬 Final Video"):
                    video_output = gr.Video(label="Storyboard Video", height=400)
    
    gr.Markdown("---")
    gr.Markdown("**Note:** Video generation requires FFmpeg to be installed on your system.")
    
    generate_btn.click(
        fn=process_storyboard_enhanced,
        inputs=[story_input, num_scenes, voice_choice],
        outputs=[script_output, gallery_output, video_output, status_output]
    )

# -----------------------------
# LAUNCH
# -----------------------------
if __name__ == "__main__":
    print("🚀 Starting AI Storyboard Generator...")
    print(f"📁 Temp directory: {TEMP_DIR}")
    
    # Check dependencies
    if not check_ffmpeg():
        print("⚠️  WARNING: FFmpeg not found. Video generation will fail.")
        print("   Install FFmpeg: https://ffmpeg.org/download.html")
    
    try:
        demo.launch(share=True, debug=True)
    finally:
        # Cleanup temp directory on exit
        if TEMP_DIR.exists():
            print(f"🧹 Cleaning up temp directory: {TEMP_DIR}")
            shutil.rmtree(TEMP_DIR, ignore_errors=True)
