import React, { useCallback, useState } from "react";
import { CanvasSegmentation } from "./components/CanvasSegmentation";
import { SegmentationToolbar } from "./components/SegmentationToolbar";

const DEFAULT_IMAGE = "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=1024";

export default function App(): React.ReactElement {
  const [imageSrc, setImageSrc] = useState<string>(DEFAULT_IMAGE);

  const handleChangeImage = useCallback((value: string) => {
    setImageSrc(value);
  }, []);

  return (
    <main style={styles.page}>
      <section style={styles.panel}>
        <SegmentationToolbar imageSrc={imageSrc} onChangeImage={handleChangeImage} />
        <CanvasSegmentation imageSrc={imageSrc} />
      </section>
    </main>
  );
}

const styles = {
  page: {
    minHeight: "100vh",
    margin: 0,
    backgroundColor: "#111827",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: "2rem",
    boxSizing: "border-box" as const,
  },
  panel: {
    width: "min(900px, 100%)",
    display: "grid",
    gap: "1.5rem",
    backgroundColor: "#ffffff",
    padding: "1.5rem",
    borderRadius: "1.25rem",
    boxShadow: "0 20px 45px rgba(15, 23, 42, 0.25)",
  },
} satisfies Record<string, React.CSSProperties>;
