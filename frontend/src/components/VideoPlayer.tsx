import React, { forwardRef } from "react";

interface VideoPlayerProps {
  src: string;
  onLoadedMetadata?: () => void;
  style?: React.CSSProperties;
  poster?: string;
}

const VideoPlayer = forwardRef<HTMLVideoElement, VideoPlayerProps>(
  ({ src, onLoadedMetadata, style, poster }, ref) => {
    return (
      <video
        ref={ref}
        src={src}
        className="video-player"
        controls
        style={style}
        poster={poster}
        preload="metadata"
        onLoadedMetadata={onLoadedMetadata}
        crossOrigin="anonymous"
      />
    );
  }
);

VideoPlayer.displayName = "VideoPlayer";

export default VideoPlayer;
