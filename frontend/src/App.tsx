import React, { useRef, useState } from "react";
import VideoPlayer from "./components/VideoPlayer";
import { videoUrl } from "./consts";

export interface FaceDetection {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  label?: string;
}

const App: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [currentVideoUrl, setCurrentVideoUrl] = useState<string>(videoUrl);
  const [processedVideoUrl, setProcessedVideoUrl] = useState<string>("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string>("");
  const [showUrlInput, setShowUrlInput] = useState(false);
  const [urlInput, setUrlInput] = useState<string>("");

  const processVideo = async () => {
    setIsProcessing(true);
    setError("");

    try {
      const res = await fetch(
        `http://127.0.0.1:8080/processed-video?src=${encodeURIComponent(
          currentVideoUrl
        )}`
      );
      if (!res.ok) {
        throw new Error("Failed to process video");
      }

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      setProcessedVideoUrl(url);
    } catch (error) {
      console.error("Error processing video:", error);
      setError("Failed to process video. Please try again.");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleChangeVideo = () => {
    setShowUrlInput(true);
    setUrlInput("");
  };

  const handleUrlSubmit = () => {
    if (urlInput.trim()) {
      setCurrentVideoUrl(urlInput.trim());
      setProcessedVideoUrl(""); // Reset processed video
      setShowUrlInput(false);
      setUrlInput("");
    }
  };

  const handleUrlCancel = () => {
    setShowUrlInput(false);
    setUrlInput("");
  };

  return (
    <div className="app-container">
      <div className="content-wrapper">
        <h1 className="title">Video Background Processor</h1>

        <div className="video-section">
          <div className="video-container original">
            <div className="video-header">
              <h2>Original Video</h2>
              <button
                onClick={handleChangeVideo}
                className="change-video-button"
                disabled={isProcessing}
              >
                Change Video
              </button>
            </div>
            <VideoPlayer
              ref={videoRef}
              src={currentVideoUrl}
              key={currentVideoUrl} // Force re-render when URL changes
              onLoadedMetadata={() => console.log("Video loaded")}
            />
          </div>

          {(isProcessing || processedVideoUrl) && (
            <div
              className={`video-container processed ${
                isProcessing ? "loading" : ""
              } ${processedVideoUrl ? "done" : ""}`}
            >
              <h2>Processed Video</h2>
              <VideoPlayer
                src={processedVideoUrl || currentVideoUrl}
                key={processedVideoUrl || currentVideoUrl} // Force re-render when URL changes
                onLoadedMetadata={() => console.log("Processed video loaded")}
                style={{
                  opacity: processedVideoUrl ? 1 : 0.3,
                  transition: "opacity 0.5s ease-out",
                }}
              />
            </div>
          )}
        </div>

        <div className="controls-section">
          <button
            onClick={processVideo}
            className={`process-button ${isProcessing ? "processing" : ""}`}
            disabled={isProcessing}
          >
            {isProcessing ? (
              <>
                <span className="spinner"></span>
                Processing...
              </>
            ) : (
              "Make Background B&W"
            )}
          </button>

          {error && <div className="error-message">{error}</div>}
        </div>

        {showUrlInput && (
          <div className="url-modal">
            <div className="url-modal-content">
              <h3>Enter Video URL</h3>
              <p>Paste a YouTube link or direct video URL:</p>
              <input
                type="text"
                value={urlInput}
                onChange={(e) => setUrlInput(e.target.value)}
                placeholder="https://www.youtube.com/watch?v=... or direct video URL"
                className="url-input"
                autoFocus
              />
              <div className="url-modal-buttons">
                <button onClick={handleUrlCancel} className="cancel-button">
                  Cancel
                </button>
                <button
                  onClick={handleUrlSubmit}
                  className="submit-button"
                  disabled={!urlInput.trim()}
                >
                  Load Video
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
