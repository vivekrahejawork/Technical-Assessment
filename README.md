# Video Background Processor

A full-stack application that processes videos to create a colorized person with black & white background effect. Supports YouTube videos and direct video URLs with advanced person segmentation using MediaPipe.

## Project Structure

This repository contains two main components:

### Backend (`/backend`)

- **Technology**: Python Flask with MediaPipe, OpenCV, and yt-dlp
- **Purpose**: Video processing, person segmentation, and YouTube video download
- **Key Files**:
  - `main.py` - Main Flask application with video processing endpoints
  - `helpers.py` - Utility functions
  - `cache/` - Video cache directory for processed files

### Frontend (`/frontend`)

- **Technology**: React with TypeScript
- **Purpose**: Modern UI for video upload, processing, and comparison
- **Key Features**:
  - YouTube URL support
  - Side-by-side video comparison
  - Loading animations and shimmer effects
  - Responsive design

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup

1. Navigate to the backend directory:

   ```bash
   cd backend
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Install yt-dlp for YouTube support:

   ```bash
   pip install yt-dlp
   ```

5. Start the backend server:
   ```bash
   python main.py
   ```

The backend will run on `http://127.0.0.1:8080`

### Cache Management

The backend creates a cache directory to store downloaded and processed videos. If you encounter issues or want to clear the cache:

```bash
# Clear and recreate cache directory
rm -rf cache && mkdir -p cache && ls -la cache

# Or manually delete specific cached files
rm cache/*.mp4
```

### Frontend Setup

1. Navigate to the frontend directory:

   ```bash
   cd frontend
   ```

2. Install Node.js dependencies:

   ```bash
   npm install
   ```

3. Start the React development server:
   ```bash
   npm start
   ```

The frontend will run on `http://localhost:3000`

## API Endpoints

### Backend Routes

- `GET /hello-world` - Test endpoint to verify backend connectivity
- `GET /processed-video?src=<video_url>` - Process video with person segmentation
- `GET /preview?src=<video_url>` - MJPEG preview stream of processed video

## Usage

1. Start both the backend and frontend servers
2. Open your browser to `http://localhost:3000`
3. Use the default video or click "Change Video" to add a YouTube URL
4. Click "Make Background B&W" to process the video
5. Watch the shimmer loading effect and compare the results side-by-side

### Supported Video Sources

- **YouTube URLs**: Any public YouTube video
- **Direct Video URLs**: Direct links to .mp4 files
- **Default Video**: Pre-configured sample video

## Features

- **Advanced Person Segmentation**: Uses MediaPipe for accurate person detection
- **Binary Separation**: Clean color/B&W separation with minimal halo
- **Audio Preservation**: Maintains original audio track in processed videos
- **Caching System**: Efficient video caching for faster repeat processing
- **Modern UI**: Responsive design with loading animations

## Technologies Used

- **Backend**: Python Flask, MediaPipe, OpenCV, yt-dlp, ffmpeg
- **Frontend**: React, TypeScript, HTML5 Video
- **Styling**: CSS3 with modern responsive design and animations
- **Video Processing**: MediaPipe Selfie Segmentation with pose detection
- **Audio Processing**: ffmpeg for audio stream copying

## Troubleshooting

### Common Issues

1. **Video not loading**: Check if the video URL is accessible and the backend is running
2. **Processing fails**: Clear the cache and restart the backend
3. **YouTube download fails**: Ensure yt-dlp is installed and the video is public
4. **Audio missing**: Check that ffmpeg is properly installed on your system

### Cache Issues

If you encounter video processing issues, clear the cache:

```bash
cd backend
rm -rf cache && mkdir -p cache && ls -la cache
```

### Dependencies

Make sure all dependencies are installed:

```bash
# Backend dependencies
cd backend
pip install -r requirements.txt
pip install yt-dlp

# Frontend dependencies
cd frontend
npm install
```

## Development

This project demonstrates advanced video processing techniques including:

- Real-time person segmentation
- MediaPipe integration
- Binary image processing
- Audio stream handling
- Modern React patterns
- Responsive UI design

## License

This project is designed for technical demonstration purposes.
