import React from "react";

interface SegmentationToolbarProps {
  readonly imageSrc: string;
  readonly onChangeImage?: (src: string) => void;
}

export const SegmentationToolbar: React.FC<SegmentationToolbarProps> = ({ imageSrc, onChangeImage }) => {
  const handleChange: React.ChangeEventHandler<HTMLInputElement> = (event) => {
    if (!onChangeImage) return;
    const value = event.target.value.trim();
    if (value.length === 0) return;
    onChangeImage(value);
  };

  return (
    <div style={toolbarStyles.container}>
      <div>
        <strong>Segment Anything 2</strong>
        <p style={toolbarStyles.instructions}>
          Click to add positive points. Hold <kbd>Shift</kbd> and click to add negative points.
        </p>
      </div>
      {onChangeImage && (
        <label style={toolbarStyles.inputLabel}>
          Image URL
          <input type="url" defaultValue={imageSrc} onBlur={handleChange} placeholder="https://example.com/image.jpg" style={toolbarStyles.input} />
        </label>
      )}
    </div>
  );
};

const toolbarStyles = {
  container: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: "1rem",
    padding: "0.75rem 1rem",
    borderRadius: "0.75rem",
    backgroundColor: "#f3f4f6",
    border: "1px solid #e5e7eb",
  } satisfies React.CSSProperties,
  instructions: {
    margin: 0,
    color: "#374151",
    fontSize: "0.875rem",
  } satisfies React.CSSProperties,
  inputLabel: {
    display: "flex",
    flexDirection: "column" as const,
    gap: "0.25rem",
    fontSize: "0.75rem",
    color: "#374151",
  } satisfies React.CSSProperties,
  input: {
    minWidth: "16rem",
    padding: "0.5rem 0.75rem",
    borderRadius: "0.5rem",
    border: "1px solid #d1d5db",
    fontSize: "0.875rem",
  } satisfies React.CSSProperties,
};
