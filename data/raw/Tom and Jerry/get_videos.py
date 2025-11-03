import yt_dlp
import os

# Folder where videos will be saved
output_folder = "videos/train"
os.makedirs(output_folder, exist_ok=True)

# List of video URLs (Tom & Jerry videos)
urls = [
    # "https://youtu.be/BS5BrXQNEtE?si=itLhiSP1ceAZg-QW",
    # "https://youtu.be/Gkysb_8N9os?si=cg0-sNwN3uf4hJwO",
    # "https://youtu.be/xdmdyRREy3Q?si=9BomDloxPgHnOEt9",
    # "https://youtu.be/8KiYj1NrmiI?si=hOCKcp3NKVDQmZP4",
    # "https://youtu.be/zqpNT-OfD7E?si=97fq03vaftDUvV2F",
    "https://youtu.be/rilFfbm7j8k?si=OThFIM2fphSUJSow",
    "https://youtu.be/di3daaObdug?si=QDCqGNGF9X2jllhr"
]

def download_videos(urls, output_path):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'quiet': False,  # show progress
        'merge_output_format': 'mp4'
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(urls)

if __name__ == "__main__":
    download_videos(urls, output_folder)
    print("All videos downloaded successfully!")
