import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import torch
from transformers import pipeline
import edge_tts
import asyncio
from vosk import Model, KaldiRecognizer
import numpy as np
import wave
from pydub import AudioSegment
import io
from datetime import datetime
import logging
import tempfile
import os
import ffmpeg
import requests
import zipfile
from typing import Optional, Tuple
import time
import shutil
from pathlib import Path
import sys
import subprocess
import platform

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"
VOSK_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
SESSION_TIMEOUT = 3600  # 1 hour
MESSAGE_LIMIT = 100
RETRY_LIMIT = 3
MODEL_DOWNLOAD_CHUNK_SIZE = 8192

class FFmpegInstaller:
    @staticmethod
    def get_os_type():
        """Determine the operating system."""
        if platform.system() == "Windows":
            return "windows"
        elif platform.system() == "Darwin":
            return "macos"
        elif platform.system() == "Linux":
            return "linux"
        return "unknown"

    @staticmethod
    def install_ffmpeg():
        """Install FFmpeg based on the operating system."""
        os_type = FFmpegInstaller.get_os_type()
        try:
            if os_type == "windows":
                # Using chocolatey for Windows
                st.info("Installing FFmpeg using Chocolatey...")
                subprocess.run(["choco", "install", "ffmpeg", "-y"], check=True)
            elif os_type == "macos":
                # Using homebrew for MacOS
                st.info("Installing FFmpeg using Homebrew...")
                subprocess.run(["brew", "install", "ffmpeg"], check=True)
            elif os_type == "linux":
                # Using apt for Linux (Ubuntu/Debian)
                st.info("Installing FFmpeg using apt...")
                subprocess.run(["sudo", "apt-get", "update"], check=True)
                subprocess.run(["sudo", "apt-get", "install", "-y", "ffmpeg"], check=True)
            else:
                st.error("Unsupported operating system for automatic FFmpeg installation.")
                return False
            return True
        except subprocess.SubprocessError as e:
            logger.error(f"FFmpeg installation error: {str(e)}")
            return False

    @staticmethod
    def check_and_install_ffmpeg():
        """Check if FFmpeg is installed and attempt to install if not."""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            st.warning("FFmpeg not found. Attempting to install...")
            if FFmpegInstaller.install_ffmpeg():
                st.success("FFmpeg installed successfully!")
                return True
            else:
                st.error("""
                Failed to install FFmpeg automatically. Please install manually:
                - Windows: Install Chocolatey and run 'choco install ffmpeg'
                - MacOS: Install Homebrew and run 'brew install ffmpeg'
                - Linux: Run 'sudo apt-get install ffmpeg'
                """)
                return False

[Previous ModelDownloader, ModelManager, AudioProcessor, and other classes remain the same...]

def check_system_requirements() -> bool:
    """Check if all system requirements are met."""
    try:
        # Check ffmpeg
        if not FFmpegInstaller.check_and_install_ffmpeg():
            return False
        
        # Check Python version
        if sys.version_info < (3, 7):
            st.error("Python 3.7 or higher is required.")
            return False
            
        # Check available disk space
        free_space = shutil.disk_usage('/').free
        if free_space < 1_000_000_000:  # 1 GB
            st.warning("Low disk space. This may affect performance.")
            
        return True
        
    except Exception as e:
        logger.error(f"System check error: {str(e)}")
        return False

[Rest of the code remains the same...]

if __name__ == "__main__":
    main()
