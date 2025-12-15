import yt_dlp
import os

# Folder where videos will be saved
output_folder = "videos"
os.makedirs(output_folder, exist_ok=True)

video_map = {
    "https://youtu.be/Ph7iu3xlpFY?si=8QddqzyuPFlv3L0D": "vol_1",
    "https://youtu.be/uaEQYtaJol4?si=GyF10J06X-MV652x": "vol_2",
    "https://youtu.be/ED4jrBs2WEI?si=4C_AcTtlZm1ptEBF": "vol_3",
    "https://youtu.be/I0rULIYGIn8?si=SO_rxgCdZaN1LYEx": "vol_4",
    "https://youtu.be/9fYNWfw1OcQ?si=LlX-c8Tja120D37e": "vol_5",
    "https://youtu.be/ydKyCjI3pQ0?si=xIKRDsRwzioahLnZ": "vol_6",
    "https://youtu.be/_GE6zf_hH48?si=MeI7ZnY70RPOgwsP": "vol_7",
    "https://youtu.be/mUdfCKtVWMo?si=eyu7gTMGlHDmN5EI": "vol_8",
    "https://youtu.be/AGD1moY4DZQ?si=4atOMYj9IvSi7_XH": "vol_9",
    # vol 10 skipped (Spike series)
    "https://youtu.be/orf5ZtQjxm0?si=SBrYnv56kZveSJTt": "vol_11",
    "https://youtu.be/pEl3-0GHyoQ?si=I2ar5evFds1QmDF6": "vol_12",
    "https://youtu.be/sHDr5iTf9jQ?si=miTs5SLcZ62EdANp": "vol_13",
    "https://youtu.be/rp7WE3_0Uag?si=9JkAzfTwMHLpw2DI": "vol_14",
    "https://youtu.be/goAdPA6D4Ko?si=vuz3cSrgyOPDm-A4": "vol_15",
    "https://youtu.be/xP4Rrc5i4Mg?si=F1nJYirQcVVozrtk": "vol_16",
    "https://youtu.be/oLr04kNjMgA?si=GOLF-AVWRyula8ne": "vol_17",
    "https://youtu.be/WhFbFAnAEwU?si=emZ2aUcRD6XHYvUq": "vol_18",
    "https://youtu.be/LrMKrnowKcc?si=iTN7PDQ0DXfZFNuC": "vol_19",
}

urls = list(video_map.keys())

def download_videos(video_map, output_path):
    for url, simple_title in video_map.items():
        print(f"\nDownloading {simple_title} ...")

        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
            'outtmpl': os.path.join(output_path, f'{simple_title}.%(ext)s'),
            'quiet': False,
            'merge_output_format': 'mp4',
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

if __name__ == "__main__":
    download_videos(video_map, output_folder)
    print("\nAll videos downloaded successfully!")
