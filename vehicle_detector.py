"""
Professional Vehicle Detection System for Moving Camera Scenarios
Supports: YOLOv8, YOLOv9, YOLOv5, Faster R-CNN, DETR
Handles: Motion blur, varying lighting, multiple vehicle types
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Model imports
try:
    from ultralytics import YOLO  # YOLOv8/v9
except ImportError:
    print("Installing ultralytics for YOLOv8/v9...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'ultralytics'])
    from ultralytics import YOLO

try:
    import torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
except ImportError:
    print("Installing torchvision for Faster R-CNN...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'torchvision'])
    import torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights


class ModelType(Enum):
    """Available detection models"""
    YOLOV8 = "yolov8"
    YOLOV9 = "yolov9"
    YOLOV5 = "yolov5"
    FASTER_RCNN = "faster_rcnn"
    DETR = "detr"


class ModelSize(Enum):
    """Model size variants"""
    NANO = "n"
    SMALL = "s"
    MEDIUM = "m"
    LARGE = "l"
    XLARGE = "x"


@dataclass
class DetectionResult:
    """Structure for detection results"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_name: str
    class_id: int


class PreProcessor:
    """Advanced preprocessing for motion blur and lighting variations"""
    
    @staticmethod
    def enhance_frame(frame: np.ndarray, denoise: bool = True, 
                     enhance_contrast: bool = True, sharpen: bool = True) -> np.ndarray:
        """
        Apply preprocessing to handle motion blur and lighting issues
        """
        processed = frame.copy()
        
        # Denoise to reduce motion blur artifacts
        if denoise:
            processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
        
        # Enhance contrast for varying lighting conditions
        if enhance_contrast:
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            processed = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
        # Sharpen to counteract motion blur
        if sharpen:
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            processed = cv2.filter2D(processed, -1, kernel)
        
        return processed
    
    @staticmethod
    def apply_motion_compensation(frame: np.ndarray, prev_frame: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply motion compensation for moving camera scenarios
        """
        if prev_frame is None:
            return frame
        
        # Convert to grayscale for optical flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow to detect camera motion
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 
                                           0.5, 3, 15, 3, 5, 1.2, 0)
        
        return frame  # Return original for now, flow can be used for tracking


class YOLODetector:
    """YOLO-based detection (v5, v8, v9)"""
    
    def __init__(self, model_type: ModelType = ModelType.YOLOV8, 
                 model_size: ModelSize = ModelSize.XLARGE):
        
        self.model_type = model_type
        self.model_size = model_size
        
        # Vehicle class IDs in COCO dataset
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        
        print(f"Loading {model_type.value}{model_size.value} model...")
        
        if model_type == ModelType.YOLOV8:
            self.model = YOLO(f'yolov8{model_size.value}.pt')
        elif model_type == ModelType.YOLOV9:
            self.model = YOLO(f'yolov9{model_size.value}.pt')
        elif model_type == ModelType.YOLOV5:
            self.model = YOLO(f'yolov5{model_size.value}u.pt')
        
        print(f"✓ Model loaded successfully")
    
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.25) -> List[DetectionResult]:
        """Detect vehicles in frame"""
        
        results = self.model(frame, conf=conf_threshold, 
                            classes=list(self.vehicle_classes.keys()),
                            verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                detections.append(DetectionResult(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    class_name=self.vehicle_classes[class_id],
                    class_id=class_id
                ))
        
        return detections


class FasterRCNNDetector:
    """Faster R-CNN detector - Higher accuracy, slightly slower"""
    
    def __init__(self):
        print("Loading Faster R-CNN model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pretrained model
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights)
        self.model.to(self.device)
        self.model.eval()
        
        # COCO vehicle classes
        self.vehicle_classes = {
            3: 'car',
            4: 'motorcycle',
            6: 'bus',
            8: 'truck'
        }
        
        print(f"✓ Faster R-CNN loaded on {self.device}")
    
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[DetectionResult]:
        """Detect vehicles using Faster R-CNN"""
        
        # Preprocess
        img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Detect
        with torch.no_grad():
            predictions = self.model(img_tensor)
        
        detections = []
        pred = predictions[0]
        
        for i in range(len(pred['boxes'])):
            score = pred['scores'][i].cpu().numpy()
            class_id = int(pred['labels'][i].cpu().numpy())
            
            if score >= conf_threshold and class_id in self.vehicle_classes:
                box = pred['boxes'][i].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)
                
                detections.append(DetectionResult(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(score),
                    class_name=self.vehicle_classes[class_id],
                    class_id=class_id
                ))
        
        return detections


class DETRDetector:
    """DETR (Detection Transformer) - Modern transformer-based approach"""
    
    def __init__(self):
        print("Loading DETR model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            from transformers import DetrImageProcessor, DetrForObjectDetection
        except ImportError:
            print("Installing transformers for DETR...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'transformers'])
            from transformers import DetrImageProcessor, DetrForObjectDetection
        
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.model.to(self.device)
        self.model.eval()
        
        # COCO vehicle classes
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        
        print(f"✓ DETR loaded on {self.device}")
    
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.7) -> List[DetectionResult]:
        """Detect vehicles using DETR"""
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        inputs = self.processor(images=image_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Detect
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process
        target_sizes = torch.tensor([frame.shape[:2]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=conf_threshold
        )[0]
        
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            class_id = int(label.cpu().numpy())
            
            if class_id in self.vehicle_classes:
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                
                detections.append(DetectionResult(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(score.cpu().numpy()),
                    class_name=self.vehicle_classes[class_id],
                    class_id=class_id
                ))
        
        return detections


class VehicleDetectionSystem:
    """
    Main detection system with all models and advanced features
    """
    
    def __init__(self, model_type: ModelType = ModelType.YOLOV8,
                 model_size: ModelSize = ModelSize.XLARGE,
                 enable_preprocessing: bool = True,
                 enable_tracking: bool = True):
        
        self.model_type = model_type
        self.enable_preprocessing = enable_preprocessing
        self.enable_tracking = enable_tracking
        
        # Initialize preprocessor
        self.preprocessor = PreProcessor()
        
        # Initialize detector based on model type
        if model_type in [ModelType.YOLOV8, ModelType.YOLOV9, ModelType.YOLOV5]:
            self.detector = YOLODetector(model_type, model_size)
        elif model_type == ModelType.FASTER_RCNN:
            self.detector = FasterRCNNDetector()
        elif model_type == ModelType.DETR:
            self.detector = DETRDetector()
        
        # Tracking variables
        self.prev_frame = None
        self.vehicle_count = 0
        
        # Performance metrics
        self.fps_list = []
    
    def process_frame(self, frame: np.ndarray, conf_threshold: float = 0.3) -> Tuple[np.ndarray, List[DetectionResult], Dict]:
        """
        Process a single frame with detection and preprocessing
        """
        start_time = time.time()
        
        # Preprocess frame
        processed_frame = frame.copy()
        if self.enable_preprocessing:
            processed_frame = self.preprocessor.enhance_frame(
                processed_frame, 
                denoise=True, 
                enhance_contrast=True, 
                sharpen=True
            )
        
        # Detect vehicles
        detections = self.detector.detect(processed_frame, conf_threshold)
        
        # Update tracking
        self.prev_frame = frame
        
        # Calculate FPS
        fps = 1.0 / (time.time() - start_time)
        self.fps_list.append(fps)
        
        # Prepare metadata
        metadata = {
            'fps': fps,
            'avg_fps': np.mean(self.fps_list[-30:]) if self.fps_list else 0,
            'num_detections': len(detections),
            'preprocessing': self.enable_preprocessing
        }
        
        return processed_frame, detections, metadata
    
    def draw_detections(self, frame: np.ndarray, detections: List[DetectionResult], 
                       metadata: Dict) -> np.ndarray:
        """
        Draw bounding boxes and information on frame
        """
        annotated = frame.copy()
        
        # Color map for different vehicle types
        color_map = {
            'car': (0, 255, 0),      # Green
            'truck': (0, 165, 255),  # Orange
            'bus': (0, 0, 255),      # Red
            'motorcycle': (255, 0, 255)  # Magenta
        }
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = color_map.get(det.class_name, (0, 255, 0))
            
            # Draw bounding box with thickness based on confidence
            thickness = max(2, int(det.confidence * 4))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label
            label = f"{det.class_name}: {det.confidence:.2f}"
            
            # Draw label background
            font_scale = 0.6
            font_thickness = 2
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            # Draw filled rectangle for label
            cv2.rectangle(annotated, 
                         (x1, y1 - label_height - 10),
                         (x1 + label_width + 10, y1),
                         color, -1)
            
            # Draw label text
            cv2.putText(annotated, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
        
        # Draw info panel
        self._draw_info_panel(annotated, metadata, detections)
        
        return annotated
    
    def _draw_info_panel(self, frame: np.ndarray, metadata: Dict, 
                        detections: List[DetectionResult]) -> None:
        """Draw information panel on frame"""
        
        h, w = frame.shape[:2]
        panel_height = 150
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Info text
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (255, 255, 255)
        thickness = 2
        
        info_lines = [
            f"Model: {self.model_type.value.upper()}",
            f"FPS: {metadata['avg_fps']:.1f} | Detections: {metadata['num_detections']}",
            f"Preprocessing: {'ON' if metadata['preprocessing'] else 'OFF'}",
        ]
        
        # Count by type
        type_counts = {}
        for det in detections:
            type_counts[det.class_name] = type_counts.get(det.class_name, 0) + 1
        
        if type_counts:
            counts_str = " | ".join([f"{k}: {v}" for k, v in type_counts.items()])
            info_lines.append(f"Vehicles: {counts_str}")
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, y_offset + i * 30),
                       font, font_scale, color, thickness)
    
    def process_video(self, video_path: str, output_path: Optional[str] = None,
                     conf_threshold: float = 0.3, show_live: bool = True,
                     skip_frames: int = 0, max_frames: Optional[int] = None) -> Dict:
        """
        Process entire video with vehicle detection
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            conf_threshold: Confidence threshold for detections
            show_live: Show live preview window
            skip_frames: Process every Nth frame (0 = process all)
            max_frames: Maximum number of frames to process
        """
        
        print(f"\n{'='*60}")
        print(f"VEHICLE DETECTION SYSTEM")
        print(f"{'='*60}")
        print(f"Model: {self.model_type.value.upper()}")
        print(f"Video: {video_path}")
        print(f"Preprocessing: {'Enabled' if self.enable_preprocessing else 'Disabled'}")
        print(f"{'='*60}\n")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video Info: {width}x{height} @ {fps}fps | Total frames: {total_frames}\n")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Output will be saved to: {output_path}\n")
        
        # Processing stats
        frame_count = 0
        processed_count = 0
        total_detections = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames if specified
                if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                    continue
                
                # Max frames limit
                if max_frames and processed_count >= max_frames:
                    break
                
                # Process frame
                processed_frame, detections, metadata = self.process_frame(frame, conf_threshold)
                
                # Draw detections
                annotated_frame = self.draw_detections(frame, detections, metadata)
                
                # Update stats
                processed_count += 1
                total_detections += len(detections)
                
                # Show live preview
                if show_live:
                    display_frame = cv2.resize(annotated_frame, (1280, 720))
                    cv2.imshow('Vehicle Detection System', display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n[!] User interrupted")
                        break
                    elif key == ord('p'):
                        print("[*] Paused - Press any key to continue")
                        cv2.waitKey(0)
                
                # Write to output
                if writer:
                    writer.write(annotated_frame)
                
                # Progress update
                if processed_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    avg_fps = np.mean(self.fps_list[-30:])
                    print(f"Progress: {progress:.1f}% | Frame: {frame_count}/{total_frames} | "
                          f"FPS: {avg_fps:.1f} | Vehicles: {len(detections)}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
        
        # Final statistics
        avg_fps = np.mean(self.fps_list) if self.fps_list else 0
        stats = {
            'total_frames': frame_count,
            'processed_frames': processed_count,
            'total_detections': total_detections,
            'avg_detections_per_frame': total_detections / processed_count if processed_count > 0 else 0,
            'avg_fps': avg_fps,
            'processing_time': processed_count / avg_fps if avg_fps > 0 else 0
        }
        
        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Processed frames: {processed_count}/{frame_count}")
        print(f"Total detections: {total_detections}")
        print(f"Avg detections/frame: {stats['avg_detections_per_frame']:.2f}")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Total processing time: {stats['processing_time']:.2f}s")
        print(f"{'='*60}\n")
        
        return stats


def main():
    """
    Main function - OPTIMIZED FOR SPEED
    """
    
    # ============================================
    # SPEED-OPTIMIZED CONFIGURATION
    # ============================================
    
    # Model settings
    MODEL = ModelType.YOLOV8
    SIZE = ModelSize.MEDIUM  # ← MEDIUM model for balanced speed/accuracy
    
    # Video settings
    VIDEO_PATH = "Your Input Video "
    OUTPUT_PATH = ""
    
    # Detection settings
    CONFIDENCE = 0.35  # Slightly higher to reduce false positives
    PREPROCESSING = False  # ← DISABLED for speed boost
    SHOW_LIVE = True
    
    # Processing limits
    PROCESS_FIRST_N_SECONDS = 120  # ← Process only first 30 seconds
    VIDEO_FPS = 30  # Your video FPS
    MAX_FRAMES = PROCESS_FIRST_N_SECONDS * VIDEO_FPS  # = 900 frames
    
    # ============================================
    
    print("\n" + "="*60)
    print("SPEED-OPTIMIZED VEHICLE DETECTION")
    print("="*60)
    print(f"Processing: First {PROCESS_FIRST_N_SECONDS} seconds")
    print(f"Model: YOLOv8 {SIZE.value.upper()}")
    print(f"Max frames: {MAX_FRAMES}")
    print("="*60 + "\n")
    
    # Initialize detection system
    detector = VehicleDetectionSystem(
        model_type=MODEL,
        model_size=SIZE,
        enable_preprocessing=PREPROCESSING
    )
    
    # Process video with speed optimizations
    stats = detector.process_video(
        video_path=VIDEO_PATH,
        output_path=OUTPUT_PATH,
        conf_threshold=CONFIDENCE,
        show_live=SHOW_LIVE,
        skip_frames=0,      # Process all frames (change to 1 for 2x speed)
        max_frames=MAX_FRAMES  # ← Only process first 30 seconds
    )
    
    print("\n" + "="*60)
    print("✓ DETECTION COMPLETE!")
    print("="*60)
    print(f"✓ Processed: {stats['processed_frames']} frames ({PROCESS_FIRST_N_SECONDS}s)")
    print(f"✓ Average FPS: {stats['avg_fps']:.2f}")
    print(f"✓ Total detections: {stats['total_detections']}")
    print(f"✓ Output saved to: {OUTPUT_PATH}")
    print("="*60)


if __name__ == "__main__":
    main()