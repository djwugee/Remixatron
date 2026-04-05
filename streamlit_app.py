import streamlit as st
import os
import json
import tempfile
import subprocess
import urllib.parse
import bz2
from pathlib import Path
from Remixatron import InfiniteJukebox
import numpy as np
import soundfile as sf
import base64
import librosa

# --- Constants and Setup ---
REMIXATRON_DIR = Path.home() / '.remixatron'
REMIXATRON_DIR.mkdir(exist_ok=True)
BOOKMARKS_FILE = REMIXATRON_DIR / 'remixatron.global.bookmarks'

if not BOOKMARKS_FILE.exists():
    with open(BOOKMARKS_FILE, 'w') as f:
        json.dump([], f)

# --- Functions ---

def load_bookmarks():
    try:
        with open(BOOKMARKS_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return []

def save_bookmarks(bookmarks):
    with open(BOOKMARKS_FILE, 'w') as f:
        json.dump(bookmarks, f, indent=3, sort_keys=True)

def fetch_from_youtube(url, status_callback):
    status_callback("Asking YouTube for audio...")
    tmp_base = os.path.join(tempfile.gettempdir(), 'remixatron_yt')
    # Use yt-dlp to download audio and metadata
    try:
        # Get metadata first
        cmd_info = ['yt-dlp', '--dump-json', '--no-playlist', url]
        info_json = subprocess.check_output(cmd_info).decode('utf-8')
        info = json.loads(info_json)
        title = info.get('title', 'Unknown Title')
        thumbnail = info.get('thumbnail', '')

        # Download audio
        cmd_dl = ['yt-dlp', '-x', '--audio-format', 'wav',
                  '-f', 'bestaudio', '--no-playlist', '-o', tmp_base + '.%(ext)s', url]
        subprocess.check_output(cmd_dl)

        file_path = tmp_base + '.wav'
        if not os.path.exists(file_path):
             # Try other common extensions if wav failed for some reason
             for ext in ['m4a', 'webm', 'mp3']:
                 if os.path.exists(tmp_base + '.' + ext):
                     file_path = tmp_base + '.' + ext
                     break

        return file_path, {"title": title, "thumbnail": thumbnail, "url": url}
    except Exception as e:
        st.error(f"Failed to download audio: {e}")
        return None, None

def fetch_from_local(uploaded_file, status_callback):
    status_callback("Processing uploaded file...")
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    return tmp_path, {"title": uploaded_file.name, "thumbnail": "", "url": ""}

def process_audio(file_path, clusters, use_cache, status_callback):
    def remixatron_callback(percentage, message):
        status_callback(message, percentage)

    jukebox = InfiniteJukebox(file_path, clusters=clusters,
                              progress_callback=remixatron_callback,
                              start_beat=0, do_async=False)

    return jukebox

# --- UI Setup ---
st.set_page_config(page_title="Remixatron", layout="wide", page_icon="🎵")

# Custom CSS for a better look
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
    }
    .track-info {
        display: flex;
        align-items: center;
        gap: 20px;
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .track-info img {
        border-radius: 5px;
        max-height: 100px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎵 Remixatron")

# Sidebar for inputs and favorites
with st.sidebar:
    st.header("Input Settings")

    # Favorites dropdown
    bookmarks = load_bookmarks()
    if bookmarks:
        st.subheader("⭐ Favorites")
        bookmark_titles = [b['title'] for b in bookmarks]
        selected_bookmark_title = st.selectbox("Select a favorite", [""] + bookmark_titles)
        if selected_bookmark_title:
             selected_bookmark = next(b for b in bookmarks if b['title'] == selected_bookmark_title)
             # Pre-fill YouTube URL if selected
             if selected_bookmark.get('url'):
                 st.info(f"Selected: {selected_bookmark_title}")

    st.divider()

    input_type = st.radio("Choose input type", ["YouTube URL", "File Upload"])

    yt_url = ""
    uploaded_file = None
    if input_type == "YouTube URL":
        default_url = ""
        if 'selected_bookmark' in locals() and selected_bookmark_title:
             default_url = selected_bookmark.get('url', "")
        yt_url = st.text_input("YouTube URL", value=default_url, placeholder="https://www.youtube.com/watch?v=...")
    else:
        uploaded_files = st.file_uploader("Upload Audio File(s)", type=["mp3", "wav", "ogg"], accept_multiple_files=True)

    clusters = st.number_input("Number of Clusters (0 for auto)", min_value=0, value=0)
    use_cache = st.checkbox("Use Cache", value=True)

    process_btn = st.button("Process Audio", type="primary")

# --- Processing Logic ---
if process_btn:
    file_path = None
    track_info = None

    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_status(message, percentage=None):
        status_text.text(message)
        if percentage is not None:
            progress_bar.progress(int(percentage * 100))

    tracks = []
    track_info = None

    if input_type == "YouTube URL" and yt_url:
        file_path, track_info = fetch_from_youtube(yt_url, update_status)
        if file_path:
            update_status("Separating Harmonic and Percussive tracks...")
            y, sr = librosa.load(file_path, sr=None, mono=False)
            # HPSS only works on mono or we do it per channel
            if y.ndim > 1:
                y_mono = librosa.to_mono(y)
            else:
                y_mono = y

            h, p = librosa.effects.hpss(y_mono)

            # For analysis we use the full mix
            analysis_file = file_path

            tracks.append({"name": "Harmonic", "data": h, "sr": sr})
            tracks.append({"name": "Percussive", "data": p, "sr": sr})

    elif input_type == "File Upload" and uploaded_files:
        if len(uploaded_files) == 1:
            file_path, track_info = fetch_from_local(uploaded_files[0], update_status)
            if file_path:
                update_status("Separating Harmonic and Percussive tracks...")
                y, sr = librosa.load(file_path, sr=None, mono=False)
                if y.ndim > 1:
                    y_mono = librosa.to_mono(y)
                else:
                    y_mono = y
                h, p = librosa.effects.hpss(y_mono)
                analysis_file = file_path
                tracks.append({"name": "Harmonic", "data": h, "sr": sr})
                tracks.append({"name": "Percussive", "data": p, "sr": sr})
        else:
            # Multitrack upload
            update_status("Processing multiple tracks...")
            max_len = 0
            sr_main = 44100 # Default to 44.1kHz for the mix

            track_paths_to_cleanup = []
            for f in uploaded_files:
                p, info = fetch_from_local(f, update_status)
                track_paths_to_cleanup.append(p)
                y, sr = librosa.load(p, sr=sr_main, mono=False) # Resample to sr_main
                tracks.append({"name": f.name, "data": y, "sr": sr_main})
                if y.shape[-1] > max_len: max_len = y.shape[-1]

            # Create a mix for analysis
            y_mix = np.zeros(max_len)
            for t in tracks:
                y_t = t["data"]
                if y_t.ndim > 1: y_t = librosa.to_mono(y_t)
                y_mix[:len(y_t)] += y_t

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_mix:
                analysis_file = tmp_mix.name
            sf.write(analysis_file, y_mix, sr_main)
            track_info = {"title": "Multitrack Remix", "thumbnail": "", "url": ""}

            # Cleanup individual uploaded files now that they are in memory
            for p in track_paths_to_cleanup:
                if os.path.exists(p):
                    os.remove(p)

    if tracks and track_info:
        jukebox = process_audio(analysis_file, clusters, use_cache, update_status)
        st.session_state['jukebox'] = jukebox
        st.session_state['track_info'] = track_info
        st.session_state['beatmap'] = [{
            'id': b['id'],
            'start': b['start'] * 1000.0,
            'duration': b['duration'] * 1000.0,
            'segment': b['segment'],
            'cluster': b['cluster'],
            'jump_candidate': b['jump_candidates']
        } for b in jukebox.beats]
        st.session_state['play_vector'] = jukebox.play_vector

        # Convert all tracks to base64
        encoded_tracks = []
        for t in tracks:
            # Use a unique name to avoid collisions
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_track:
                tmp_track_path = tmp_track.name

            sf.write(tmp_track_path, t['data'].T, t['sr'], subtype='PCM_16')
            with open(tmp_track_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode()
            encoded_tracks.append({"name": t['name'], "base64": encoded})
            os.remove(tmp_track_path)

        st.session_state['encoded_tracks'] = encoded_tracks
        st.session_state['audio_format'] = 'wav'

        status_text.empty()
        progress_bar.empty()
        st.success("Processing complete!")

        # Final cleanup of analysis file and YouTube file if it exists
        if os.path.exists(analysis_file):
            os.remove(analysis_file)

# --- Main Area: Visualization and Playback ---
if 'jukebox' in st.session_state:
    track_info = st.session_state['track_info']

    # Display track info
    st.markdown(f"""
        <div class="track-info">
            <img src="{track_info['thumbnail'] if track_info['thumbnail'] else 'https://via.placeholder.com/100?text=No+Thumb'}" alt="Thumbnail">
            <div>
                <h3>{track_info['title']}</h3>
                <p>{track_info['url']}</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Bookmark button
    col1, col2 = st.columns([1, 5])
    with col1:
        is_bookmarked = any(b['title'] == track_info['title'] for b in bookmarks)
        if st.button("⭐ Unstar" if is_bookmarked else "⭐ Star"):
            if is_bookmarked:
                bookmarks = [b for b in bookmarks if b['title'] != track_info['title']]
            else:
                bookmarks.append({
                    "title": track_info['title'],
                    "thumbnail": track_info['thumbnail'],
                    "url": track_info['url'],
                    "clusters": st.session_state['jukebox'].clusters
                })
            save_bookmarks(bookmarks)
            st.rerun()

    st.divider()

    # Custom HTML/JS component for playback and visualization
    beatmap_json = json.dumps(st.session_state['beatmap'])
    play_vector_json = json.dumps(st.session_state['play_vector'])
    encoded_tracks_json = json.dumps(st.session_state['encoded_tracks'])

    audio_format = st.session_state.get('audio_format', 'mp3')
    html_code = f"""
    <div id="viz-container">
        <canvas id="viz" style="width: 100%; height: 350px; background-color: white; border-radius: 10px;"></canvas>
    </div>
    <div id="controls" style="margin-top: 20px; display: flex; flex-direction: column; gap: 15px; background-color: #343a40; padding: 15px; border-radius: 10px; color: white;">
        <div style="display: flex; align-items: center; gap: 20px;">
            <button id="playBtn" style="padding: 10px 20px; font-size: 20px; cursor: pointer; border-radius: 5px; border: none; background-color: #28a745; color: white;">Play</button>
            <div id="info" style="font-family: sans-serif; font-size: 16px;"></div>
        </div>
        <div id="track-volumes" style="display: flex; gap: 20px; flex-wrap: wrap;"></div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/howler/2.2.3/howler.min.js"></script>
    <script>
        const beatmap = {beatmap_json};
        const playvector = {play_vector_json};
        const encodedTracks = {encoded_tracks_json};

        let sounds = [];
        let sndIndex = 0;
        let timerPlaybackID = null;
        const canvas = document.getElementById('viz');
        const ctx = canvas.getContext('2d');
        const playBtn = document.getElementById('playBtn');
        const infoDiv = document.getElementById('info');
        const trackVolumesDiv = document.getElementById('track-volumes');

        const clusters = Math.max(...beatmap.map(b => b.cluster)) + 1;
        const segments = Math.max(...beatmap.map(b => b.segment)) + 1;

        const colorMap = [];
        for (let i = 0; i < clusters; i++) {{
            colorMap.push(rainbowStop(i / (clusters + 1)));
        }}

        function rainbowStop(h) {{
            let f = (n, k = (n + h * 12) % 12) => .5 - .5 * Math.max(Math.min(k - 3, 9 - k, 1), -1);
            let rgb2hex = (r, g, b) => "#" + [r, g, b].map(x => Math.round(x * 255).toString(16).padStart(2, 0)).join('');
            return rgb2hex(f(0), f(8), f(4));
        }}

        function initSounds() {{
            const spritedef = {{}};
            beatmap.forEach(beat => {{
                spritedef[beat.id + 1] = [beat.start, beat.duration];
            }});

            encodedTracks.forEach((track, index) => {{
                const audioData = "data:audio/{audio_format};base64," + track.base64;
                const s = new Howl({{
                    src: [audioData],
                    format: ['{audio_format}'],
                    sprite: spritedef,
                    onplay: () => {{
                        if (index === 0) {{
                            if (!timerPlaybackID) {{
                                const currentBeat = beatmap[playvector[sndIndex].beat];
                                timerPlaybackID = setTimeout(onSoundEnd, currentBeat.duration);
                            }}
                            drawViz();
                        }}
                    }}
                }});
                sounds.push(s);

                // Add volume control
                const volControl = document.createElement('div');
                volControl.innerHTML = `
                    <label style="font-size: 12px; display: block;">${{track.name}}</label>
                    <input type="range" min="0" max="1" step="0.01" value="1" style="width: 100px;"
                           oninput="sounds[${{index}}].volume(this.value)">
                `;
                trackVolumesDiv.appendChild(volControl);
            }});
        }}

        playBtn.onclick = () => {{
            if (sounds.length === 0) {{
                initSounds();
                playBtn.innerText = "Pause";
                playBtn.style.backgroundColor = "#dc3545";
                startPlayback();
            }} else if (sounds[0].playing()) {{
                sounds.forEach(s => s.pause());
                clearTimeout(timerPlaybackID);
                timerPlaybackID = null;
                playBtn.innerText = "Play";
                playBtn.style.backgroundColor = "#28a745";
            }} else {{
                sounds.forEach(s => s.play(String(playvector[sndIndex].beat + 1)));
                playBtn.innerText = "Pause";
                playBtn.style.backgroundColor = "#dc3545";
                const currentBeat = beatmap[playvector[sndIndex].beat];
                timerPlaybackID = setTimeout(onSoundEnd, currentBeat.duration);
            }}
        }};

        function startPlayback() {{
            sndIndex = 0;
            const currentBeatIdx = playvector[sndIndex].beat;
            sounds.forEach(s => s.play(String(currentBeatIdx + 1)));
        }}

        function onSoundEnd() {{
            sndIndex++;
            if (sndIndex >= playvector.length) {{
                sndIndex = 0; // Loop back or stop
            }}
            const toplay = playvector[sndIndex].beat + 1;
            sounds.forEach(s => s.play(String(toplay)));
            timerPlaybackID = setTimeout(onSoundEnd, beatmap[toplay - 1].duration);
        }}

        function drawViz() {{
            const width = canvas.offsetWidth;
            const height = canvas.offsetHeight;
            if (canvas.width !== width || canvas.height !== height) {{
                canvas.width = width;
                canvas.height = height;
            }}

            ctx.clearRect(0, 0, width, height);

            const xOffset = 5;
            const beatWidth = (width - 10) / beatmap.length;
            const timelineHeight = 150;
            const timelineY = (height / 2) - (timelineHeight / 2);

            const currentBeatIdx = playvector[sndIndex].beat;
            const currentBeat = beatmap[currentBeatIdx];
            const currentX = (currentBeatIdx * beatWidth) + xOffset;

            // Draw background timeline
            ctx.strokeStyle = '#ccc';
            ctx.strokeRect(xOffset, timelineY, width - 10, timelineHeight);
            ctx.fillStyle = '#fafafa';
            ctx.fillRect(xOffset, timelineY, width - 10, timelineHeight);

            // Draw beats
            for (let i = 0; i < beatmap.length; i++) {{
                const isSameSegment = beatmap[i].segment === currentBeat.segment;
                const isJumpCandidate = currentBeat.jump_candidate.includes(i);

                ctx.globalAlpha = (isSameSegment || isJumpCandidate) ? 1.0 : 0.15;
                ctx.fillStyle = colorMap[beatmap[i].cluster];
                ctx.fillRect((i * beatWidth) + xOffset, timelineY + 1, beatWidth + 0.5, timelineHeight - 2);
            }}

            // Draw jump arcs
            ctx.globalAlpha = 0.6;
            let over = true;
            currentBeat.jump_candidate.forEach(c => {{
                ctx.beginPath();
                ctx.strokeStyle = '#888';
                ctx.lineWidth = Math.max(1, beatWidth / 2);
                const nextX = (c * beatWidth) + xOffset;

                if (over) {{
                    ctx.moveTo(currentX, timelineY);
                    ctx.quadraticCurveTo((nextX + currentX) / 2, 0, nextX, timelineY);
                }} else {{
                    ctx.moveTo(currentX, timelineY + timelineHeight);
                    ctx.quadraticCurveTo((nextX + currentX) / 2, height, nextX, timelineY + timelineHeight);
                }}
                ctx.stroke();
                over = !over;
            }});

            // Draw marker
            ctx.globalAlpha = 1.0;
            ctx.fillStyle = 'black';
            ctx.fillRect(currentX, timelineY - 10, Math.max(2, beatWidth), timelineHeight + 20);

            // Info text
            const leftInSeq = playvector[sndIndex].seq_len - playvector[sndIndex].seq_pos;
            const ratio = (segments / clusters).toFixed(3);
            infoDiv.innerText = `Pos: ${{sndIndex}} | Beats: ${{beatmap.length}} | Clusters: ${{clusters}} | Segments: ${{segments}} | Ratio: ${{ratio}} | Next Jump: ${{leftInSeq === 0 ? '!!!' : leftInSeq}}`;

            if (sounds[0] && sounds[0].playing()) {{
                requestAnimationFrame(drawViz);
            }}
        }}

        // Handle window resize
        window.onresize = () => drawViz();
    </script>
    """

    st.iframe(html_code, height=450)
