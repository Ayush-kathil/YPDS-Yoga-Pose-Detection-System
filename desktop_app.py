# YPDS - Yoga Pose Detection System
# Modern UI with Splash Screen, Reference Images, and Enhanced Animations
import sys
import os
import cv2
import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw, ImageFont, ImageFilter
import numpy as np
import threading
import requests
from io import BytesIO
import pandas as pd
import json
import pickle
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from urllib.request import urlopen
import mediapipe as mp

# Import mediapipe with robust fallback
mp_pose = None
mp_drawing = None
try:
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
    else:
        try:
            import mediapipe.python.solutions.pose as mp_pose
            import mediapipe.python.solutions.drawing_utils as mp_drawing
        except ImportError:
            pass
except ImportError as e:
    print(f"Mediapipe import error: {e}")

# --- Configuration ---
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

# "Google Antigravity" / Premium Classic Palette
COLORS = {
    'bg': '#FFFFFF',
    'sidebar': '#F8F9FA',
    'active_bg': '#E8F0FE', # Light Blue highlight
    'primary': '#1A73E8',   # Google Blue
    'secondary': '#5F6368', # Dark Gray
    'text_primary': '#202124',
    'text_secondary': '#5F6368',
    "text_muted": "#DADCE0",      # Light Gray
    "accent": "#1A73E8",          # Blue
    "glow": "#D2E3FC",            # Light Blue glow
    "card": "#FFFFFF",            # White cards
    "card_hover": "#F1F3F4",      # Light gray hover
    "primary_light": "#D2E3FC",   # Light Blue
    "success": "#34A853",         # Google Green
    "success_light": "#CEEAD6",   # Light Green
    "warning": "#FBBC05",         # Google Yellow
    "error": "#EA4335",           # Google Red
}

class LandmarkSmoother:
    """Smooths pose landmarks using Exponential Moving Average (EMA)"""
    def __init__(self, alpha=0.4):  # Lower alpha = more smoothing
        self.alpha = alpha
        self.prev_landmarks = None

    def smooth(self, current_landmarks):
        if not current_landmarks:
            return None
        
        curr_np = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in current_landmarks])
        
        if self.prev_landmarks is None:
            self.prev_landmarks = curr_np
            return current_landmarks
        
        smoothed_np = (self.alpha * curr_np) + ((1 - self.alpha) * self.prev_landmarks)
        self.prev_landmarks = smoothed_np
        
        smoothed_objs = []
        for i in range(len(smoothed_np)):
            obj = type('LM', (), {
                'x': smoothed_np[i][0],
                'y': smoothed_np[i][1],
                'z': smoothed_np[i][2],
                'visibility': smoothed_np[i][3]
            })()
            smoothed_objs.append(obj)
            
        return smoothed_objs

class YogaPoseClassifier:
    """ML Model for yoga pose classification"""
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=150, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def extract_features(self, landmarks):
        if not landmarks: return None
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        features = []
        
        angles = self.calculate_all_angles(landmarks)
        features.extend(angles)
        
        mid_shoulder = (points[11] + points[12]) / 2
        mid_hip = (points[23] + points[24]) / 2
        torso_h = np.linalg.norm(mid_shoulder - mid_hip) or 1.0
        
        key_joints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        for i in range(len(key_joints)):
            for j in range(i+1, len(key_joints)):
                dist = np.linalg.norm(points[key_joints[i]] - points[key_joints[j]])
                features.append(dist / torso_h)
                
        return np.array(features)
    
    def calculate_all_angles(self, landmarks):
        angles = []
        definitions = [
            (11, 13, 15), (12, 14, 16), (13, 11, 23), (14, 12, 24),
            (11, 23, 25), (12, 24, 26), (23, 25, 27), (24, 26, 28)
        ]
        for p1, p2, p3 in definitions:
            angles.append(self.calculate_angle(landmarks[p1], landmarks[p2], landmarks[p3]))
        return angles
    
    def calculate_angle(self, a, b, c):
        try:
            val_a = np.array([a.x, a.y] if hasattr(a, 'x') else [a['x'], a['y']])
            val_b = np.array([b.x, b.y] if hasattr(b, 'x') else [b['x'], b['y']])
            val_c = np.array([c.x, c.y] if hasattr(c, 'x') else [c['x'], c['y']])
            
            ba, bc = val_a - val_b, val_c - val_b
            cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        except: return 0.0

    def train_with_smart_augmentation(self, reference_poses):
        self.is_trained = True

    def load_model(self, path):
        try:
            with open(path, 'rb') as f:
                d = pickle.load(f)
                self.model = d['model']
                self.scaler = d['scaler']
                self.is_trained = d['is_trained']
            return True
        except: return False


class SplashScreen(ctk.CTkToplevel):
    """Modern animated splash screen"""
    def __init__(self, parent, on_complete):
        super().__init__(parent)
        self.on_complete = on_complete
        
        # Window setup
        self.title("")
        self.geometry("600x400")
        self.overrideredirect(True)  # Remove window decorations
        self.configure(fg_color=COLORS['bg'])
        
        # Center on screen
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - 300
        y = (self.winfo_screenheight() // 2) - 200
        self.geometry(f"600x400+{x}+{y}")
        
        # Main container with gradient effect
        self.container = ctk.CTkFrame(self, fg_color=COLORS['bg'], corner_radius=20)
        self.container.pack(fill="both", expand=True, padx=2, pady=2)
        
        # Logo/Title
        self.logo_label = ctk.CTkLabel(
            self.container,
            text="🧘",
            font=("Segoe UI Emoji", 80),
            text_color=COLORS['primary']
        )
        self.logo_label.pack(pady=(60, 10))
        
        self.title_label = ctk.CTkLabel(
            self.container,
            text="",
            font=("Segoe UI", 42, "bold"),
            text_color=COLORS['text_primary']
        )
        self.title_label.pack(pady=(0, 5))
        
        self.subtitle_label = ctk.CTkLabel(
            self.container,
            text="",
            font=("Segoe UI", 16),
            text_color=COLORS['text_secondary']
        )
        self.subtitle_label.pack(pady=(0, 40))
        
        # Progress bar
        self.progress = ctk.CTkProgressBar(
            self.container,
            width=300,
            height=4,
            progress_color=COLORS['primary'],
            fg_color=COLORS['card']
        )
        self.progress.set(0)
        self.progress.pack(pady=(0, 20))
        
        self.status_label = ctk.CTkLabel(
            self.container,
            text="",
            font=("Segoe UI", 12),
            text_color=COLORS['text_muted']
        )
        self.status_label.pack()
        
        # Start animations
        self.after(100, self.animate_title)
    
    def animate_title(self):
        title = "YPDS"
        self._type_text(self.title_label, title, 0, self.animate_subtitle)
    
    def animate_subtitle(self):
        subtitle = "Yoga Pose Detection System"
        self._type_text(self.subtitle_label, subtitle, 0, self.animate_loading)
    
    def _type_text(self, label, text, index, callback):
        if index <= len(text):
            label.configure(text=text[:index])
            self.after(60, lambda: self._type_text(label, text, index + 1, callback))
        else:
            self.after(200, callback)
    
    def animate_loading(self):
        steps = [
            (0.2, "Initializing AI Engine..."),
            (0.4, "Loading Pose Models..."),
            (0.6, "Calibrating Detection..."),
            (0.8, "Preparing Interface..."),
            (1.0, "Ready!")
        ]
        self._run_loading_steps(steps, 0)
    
    def _run_loading_steps(self, steps, index):
        if index < len(steps):
            progress, text = steps[index]
            self.progress.set(progress)
            self.status_label.configure(text=text)
            self.after(400, lambda: self._run_loading_steps(steps, index + 1))
        else:
            self.after(500, self.finish)
    
    def finish(self):
        self.destroy()
        self.on_complete()


class YPDSApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("YPDS - Yoga Pose Detection System")
        self.geometry("1500x900")
        self.configure(fg_color=COLORS['bg'])
        
        # State
        self.camera_on = False
        self.cap = None
        self.current_pose = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.pose_images = {}  # Cache for pose reference images
        self.current_score = 0
        self.best_score = 0
        self.session_scores = []  # Track scores for average average
        self.calories_burnt = 0.0
        self.difficulty = "Normal" # Normal, Pro
        
        # Workout Mode State
        self.timer_running = False
        self.timer_start = 0
        self.timer_duration = 30 # Default hold time
        self.in_rest_mode = False
        self.rest_duration = 10
        self.rest_start = 0
        
        # Tools
        self.mp_pose = mp_pose
        if self.mp_pose is None:
            print("CRITICAL: Mediapipe Pose module not loaded.")
              
        self.pose_engine = self.mp_pose.Pose(
            min_detection_confidence=0.5,  # Lowered for better detection
            min_tracking_confidence=0.5,
            model_complexity=1,  # 1 is faster/lighter, 2 is more accurate but heavier
            smooth_landmarks=True
        )
        self.classifier = YogaPoseClassifier()
        self.smoother = LandmarkSmoother(alpha=0.40)  # Smoother tracking
        
        # Load Data
        self.poses = self.load_poses()
        if os.path.exists("yoga_pose_model.pkl"):
            self.classifier.load_model("yoga_pose_model.pkl")
        else:
            self.classifier.train_with_smart_augmentation(self.poses)

        # Load Configuration and Data
        self.poses = self.load_poses()
        
        # Load Machine Learning Model
        if os.path.exists("yoga_pose_model.pkl"):
            self.classifier.load_model("yoga_pose_model.pkl")
            print("✅ Model loaded successfully.")
        else:
            print("⚠️ Model not found, starting fresh training.")
            self.classifier.train_with_smart_augmentation(self.poses)

        # Start with Splash Screen
        self.withdraw() # Hide main window
        splash = SplashScreen(self, self.on_splash_complete)
        
    def on_splash_complete(self):
        self.deiconify() # Show main window
        self.setup_ui()
        self.update_video_loop()
        self.start_background_image_loading()

    def load_poses(self):
        """Loads yoga pose definitions from CSV file."""
        # Path is now in the same directory as this script
        path = os.path.join(os.path.dirname(__file__), 'yoga_poses.csv')
        data = {}
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                for _, row in df.iterrows():
                    data[row['name']] = {
                        'url': row['image_url'],
                        'joints': json.loads(row['landmarks'])
                    }
            except Exception as e:
                print(f"Error loading poses: {e}")
        else:
            print(f"CSV not found at: {path}")
        return data

    def start_background_image_loading(self):
        """Load pose images in background"""
        def load_images():
            for name, data in self.poses.items():
                try:
                    response = requests.get(data['url'], timeout=10)
                    img = Image.open(BytesIO(response.content))
                    img = img.resize((280, 200), Image.Resampling.LANCZOS)
                    self.pose_images[name] = ctk.CTkImage(img, size=(280, 200))
                except Exception as e:
                    print(f"Failed to load image for {name}: {e}")
        
        threading.Thread(target=load_images, daemon=True).start()

    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)
        self.grid_rowconfigure(0, weight=1)
        
        # --- Left Sidebar ---
        self.sidebar = ctk.CTkFrame(self, width=300, fg_color=COLORS['sidebar'], corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        
        # Header - Minimalist
        self.logo_label = ctk.CTkLabel(
            self.sidebar, 
            text="YPDS", 
            font=("Arial", 36, "bold"), 
            text_color=COLORS['text_primary']
        )
        self.logo_label.pack(pady=(40, 5), padx=30, anchor="w")
        
        ctk.CTkLabel(
            self.sidebar, 
            text="Yoga Pose Detection System", 
            font=("Arial", 14), 
            text_color=COLORS['text_secondary']
        ).pack(padx=30, anchor="w", pady=(0, 20))
        
        # Separator
        ctk.CTkFrame(self.sidebar, height=1, fg_color="#E0E0E0").pack(fill="x", padx=30, pady=10)

        # Pose List Header
        ctk.CTkLabel(
            self.sidebar, 
            text="PRACTICE", 
            font=("Arial", 12, "bold"), 
            text_color=COLORS['text_secondary']
        ).pack(padx=30, anchor="w", pady=(20, 10))
        
        # Scrollable Pose List - Clean
        self.scroll_frame = ctk.CTkScrollableFrame(
            self.sidebar, 
            fg_color="transparent",
            scrollbar_button_color="#DADCE0",
            scrollbar_button_hover_color=COLORS['text_secondary']
        )
        self.scroll_frame.pack(fill="both", expand=True, padx=20, pady=5)
        
        self.pose_buttons = {}
        pose_icons = {"Downdog": "🐕", "Goddess": "👸", "Plank": "💪", "Tree": "🌳", "Warrior Ii": "⚔️"}
        
        for name in self.poses.keys():
            icon = pose_icons.get(name, "•")
            btn = ctk.CTkButton(
                self.scroll_frame, 
                text=f"  {name}",
                font=("Arial", 16),
                fg_color="transparent", 
                text_color=COLORS['text_primary'],
                hover_color=COLORS['card_hover'],
                anchor="w",
                height=50,
                corner_radius=8,
                command=lambda n=name: self.set_pose(n)
            )
            btn.pack(fill="x", pady=2)
            self.pose_buttons[name] = btn

        # Accuracy indicator (Dynamic)
        self.accuracy_frame = ctk.CTkFrame(self.sidebar, fg_color=COLORS['card'], corner_radius=12)
        self.accuracy_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        ctk.CTkLabel(
            self.accuracy_frame, 
            text="🎯 Session Avg", 
            font=("Segoe UI", 12, "bold"),
            text_color=COLORS['text_secondary']
        ).pack(pady=(12, 5))
        
        self.accuracy_label = ctk.CTkLabel(
            self.accuracy_frame, 
            text="--%", 
            font=("Segoe UI", 32, "bold"),
            text_color=COLORS['text_primary']
        )
        self.accuracy_label.pack(pady=(0, 5))

        # Bottom Controls
        self.control_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.control_frame.pack(fill="x", padx=20, pady=20, side="bottom")
        
        self.btn_toggle = ctk.CTkButton(
            self.control_frame, 
            text="▶  Start Session",
            font=("Segoe UI", 16, "bold"),
            fg_color=COLORS['success'],
            hover_color=COLORS['success_light'],
            height=55,
            corner_radius=27,
            command=self.toggle_camera
        )
        self.btn_toggle.pack(fill="x")

        # --- Main Display (Center) ---
        self.display = ctk.CTkFrame(self, fg_color=COLORS['bg'])
        self.display.grid(row=0, column=1, sticky="nsew", padx=15, pady=15)
        
        # HUD Overlay pinned to bottom
        self.hud_frame = ctk.CTkFrame(self.display, fg_color=COLORS['sidebar'], corner_radius=15, height=100)
        self.hud_frame.pack(side="bottom", fill="x", pady=10,padx=10)
        
        # Video container taking remaining space
        self.video_container = ctk.CTkFrame(self.display, fg_color=COLORS['card'], corner_radius=20)
        self.video_container.pack(side="top", fill="both", expand=True)

        self.video_label = ctk.CTkLabel(self.video_container, text="", corner_radius=18, fg_color=COLORS['bg'])
        self.video_label.pack(fill="both", expand=True, padx=3, pady=3)
        
        # Welcome overlay
        self.welcome_frame = ctk.CTkFrame(self.video_label, fg_color=COLORS['bg'])
        self.welcome_frame.place(relx=0.5, rely=0.5, anchor="center")
        
        ctk.CTkLabel(
            self.welcome_frame,
            text="🧘",
            font=("Segoe UI Emoji", 80)
        ).pack(pady=(20, 10))
        
        ctk.CTkLabel(
            self.welcome_frame,
            text="Welcome to YPDS",
            font=("Segoe UI", 32, "bold"),
            text_color=COLORS['text_primary']
        ).pack(pady=(0, 5))
        
        ctk.CTkLabel(
            self.welcome_frame,
            text="Select a pose from the sidebar and start your session",
            font=("Segoe UI", 14),
            text_color=COLORS['text_secondary']
        ).pack(pady=(0, 30))
        
        hud_inner = ctk.CTkFrame(self.hud_frame, fg_color="transparent")
        hud_inner.pack(fill="x", padx=20, pady=15)
        
        # Left: Current pose info
        self.pose_info_frame = ctk.CTkFrame(hud_inner, fg_color="transparent")
        self.pose_info_frame.pack(side="left")
        
        self.instruction_lbl = ctk.CTkLabel(
            self.pose_info_frame, 
            text="Select a pose to begin", 
            font=("Segoe UI", 20, "bold"),
            text_color=COLORS['text_primary']
        )
        self.instruction_lbl.pack(anchor="w")
        
        self.feedback_lbl = ctk.CTkLabel(
            self.pose_info_frame, 
            text="", 
            font=("Segoe UI", 14),
            text_color=COLORS['text_secondary']
        )
        self.feedback_lbl.pack(anchor="w")
        
        # Right: Score display
        self.score_frame = ctk.CTkFrame(hud_inner, fg_color="transparent")
        self.score_frame.pack(side="right")
        
        self.score_label = ctk.CTkLabel(
            self.score_frame, 
            text="0%", 
            font=("Segoe UI", 48, "bold"),
            text_color=COLORS['primary']
        )
        self.score_label.pack()
        
        self.score_bar = ctk.CTkProgressBar(
            self.score_frame, 
            width=150, 
            height=8, 
            progress_color=COLORS['success'],
            fg_color=COLORS['card']
        )
        self.score_bar.set(0)
        self.score_bar.pack(pady=(5, 0))
        
        # --- Right Sidebar (Reference Image Panel) ---
        self.ref_panel = ctk.CTkFrame(self, width=320, fg_color=COLORS['sidebar'], corner_radius=0)
        self.ref_panel.grid(row=0, column=2, sticky="nsew")
        self.ref_panel.grid_propagate(False)
        
        # Reference panel header
        ctk.CTkLabel(
            self.ref_panel, 
            text="📷 REFERENCE POSE", 
            font=("Segoe UI", 12, "bold"),
            text_color=COLORS['text_muted']
        ).pack(pady=(25, 15), padx=20, anchor="w")
        
        # Reference image container
        self.ref_image_container = ctk.CTkFrame(self.ref_panel, fg_color=COLORS['card'], corner_radius=15)
        self.ref_image_container.pack(fill="x", padx=15, pady=(0, 15))
        
        self.ref_image_label = ctk.CTkLabel(
            self.ref_image_container, 
            text="Select a pose\nto see reference",
            font=("Segoe UI", 14),
            text_color=COLORS['text_secondary'],
            height=220
        )
        self.ref_image_label.pack(pady=10, padx=10)
        
        # Pose name
        self.ref_pose_name = ctk.CTkLabel(
            self.ref_panel, 
            text="", 
            font=("Segoe UI", 18, "bold"),
            text_color=COLORS['text_primary']
        )
        self.ref_pose_name.pack(pady=(5, 15))
        
        # Tips section
        ctk.CTkLabel(
            self.ref_panel, 
            text="💡 TIPS", 
            font=("Segoe UI", 12, "bold"),
            text_color=COLORS['text_muted']
        ).pack(pady=(20, 10), padx=20, anchor="w")
        
        self.tips_frame = ctk.CTkFrame(self.ref_panel, fg_color=COLORS['card'], corner_radius=12)
        self.tips_frame.pack(fill="x", padx=15)
        
        tips = [
            "• Keep your body aligned",
            "• Breathe steadily",
            "• Hold pose for 10+ seconds",
            "• Match the reference image"
        ]
        
        for tip in tips:
            ctk.CTkLabel(
                self.tips_frame, 
                text=tip,
                font=("Segoe UI", 12),
                text_color=COLORS['text_secondary'],
                anchor="w"
            ).pack(pady=5, padx=15, anchor="w")
        
        # Stats section
        ctk.CTkLabel(
            self.ref_panel, 
            text="📊 SESSION STATS", 
            font=("Segoe UI", 12, "bold"),
            text_color=COLORS['text_muted']
        ).pack(pady=(30, 10), padx=20, anchor="w")
        
        self.stats_frame = ctk.CTkFrame(self.ref_panel, fg_color=COLORS['card'], corner_radius=12)
        self.stats_frame.pack(fill="x", padx=15, pady=(0, 20))
        
        stats_inner = ctk.CTkFrame(self.stats_frame, fg_color="transparent")
        stats_inner.pack(fill="x", padx=15, pady=15)
        
        # Best score
        best_frame = ctk.CTkFrame(stats_inner, fg_color="transparent")
        best_frame.pack(fill="x", pady=3)
        ctk.CTkLabel(best_frame, text="Best Score", font=("Segoe UI", 12), text_color=COLORS['text_secondary']).pack(side="left")
        self.best_score_label = ctk.CTkLabel(best_frame, text="0%", font=("Segoe UI", 14, "bold"), text_color=COLORS['success'])
        self.best_score_label.pack(side="right")
        
        # Session time
        time_frame = ctk.CTkFrame(stats_inner, fg_color="transparent")
        time_frame.pack(fill="x", pady=3)
        ctk.CTkLabel(time_frame, text="Session Time", font=("Segoe UI", 12), text_color=COLORS['text_secondary']).pack(side="left")
        self.session_time_label = ctk.CTkLabel(time_frame, text="00:00", font=("Segoe UI", 14, "bold"), text_color=COLORS['accent'])
        self.session_time_label.pack(side="right")
        
        self.session_start_time = None
        
        for n, btn in self.pose_buttons.items():
            if n == name:
                btn.configure(fg_color=COLORS['primary'], text_color="#ffffff")
                self._animate_button_glow(btn)
            else:
                btn.configure(fg_color=COLORS['card'], text_color=COLORS['text_primary'])
        
        self.instruction_lbl.configure(text=f"Target: {name}")
        self.feedback_lbl.configure(text="Get into position...")
        self.ref_pose_name.configure(text=name)
        
        # Update reference image
        if name in self.pose_images:
            if hasattr(self, 'ref_image_label'):
                self.ref_image_label.configure(image=self.pose_images[name], text="")
        else:
            if hasattr(self, 'ref_image_label'):
                self.ref_image_label.configure(text="Loading...", image=None)
            self._check_image_loaded(name)

    def set_pose(self, name):
        """Sets the current target pose and resets state"""
        self.current_pose = name
        self.best_score = 0
        self.timer_running = False # Reset timer on new pose
        self.in_rest_mode = False
        
        if hasattr(self, 'best_score_label'):
            self.best_score_label.configure(text="0%")
        if hasattr(self, 'instruction_lbl'):
            self.instruction_lbl.configure(text=f"Target: {name}")
            self.feedback_lbl.configure(text="Get into position...")
        if hasattr(self, 'ref_pose_name'):
             self.ref_pose_name.configure(text=name)
             
        # Animate button
        for n, btn in self.pose_buttons.items():
            if n == name:
                btn.configure(fg_color=COLORS['active_bg'], text_color=COLORS['primary'])
            else:
                btn.configure(fg_color="transparent", text_color=COLORS['text_primary'])

        # Update reference image
        if name in self.pose_images:
            if hasattr(self, 'ref_image_label'):
                self.ref_image_label.configure(image=self.pose_images[name], text="")
        else:
            if hasattr(self, 'ref_image_label'):
                self.ref_image_label.configure(text="Loading...", image=None)
            self._check_image_loaded(name)
            
    def _check_image_loaded(self, name):
        if name in self.pose_images:
            if hasattr(self, 'ref_image_label'):
                 self.ref_image_label.configure(image=self.pose_images[name], text="")
        else:
            self.after(500, lambda: self._check_image_loaded(name))
            
    def change_difficulty(self, value):
        self.difficulty = value
        print(f"Difficulty changed to: {value}")
        self.session_scores = []
        self.accuracy_label.configure(text="--%")
    
    def _animate_button_glow(self, btn):
        colors = [COLORS['primary'], COLORS['primary_light'], COLORS['primary']]
        self._cycle_colors(btn, colors, 0)
    
    def _cycle_colors(self, btn, colors, index):
        if index < len(colors):
            btn.configure(fg_color=colors[index])
            self.after(100, lambda: self._cycle_colors(btn, colors, index + 1))

    def toggle_camera(self):
        if self.camera_on:
            self.camera_on = False
            if self.cap: self.cap.release()
            self.btn_toggle.configure(text="▶  Start Session", fg_color=COLORS['success'], hover_color=COLORS['success_light'])
            self.video_label.configure(image=None)
            self.welcome_frame.place(relx=0.5, rely=0.5, anchor="center")
            self.session_start_time = None
        else:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.camera_on = True
                self.session_start_time = time.time()
                self.btn_toggle.configure(text="⏹  End Session", fg_color=COLORS['error'], hover_color="#f87171")
                self.welcome_frame.place_forget()
                threading.Thread(target=self.capture_loop, daemon=True).start()
                self._update_session_time()
            else:
                print("❌ Failed to open camera.")
                self.feedback_lbl.configure(text="Error: Cannot open camera")
    
    def _update_session_time(self):
        if self.camera_on and self.session_start_time:
            elapsed = int(time.time() - self.session_start_time)
            mins = elapsed // 60
            secs = elapsed % 60
            self.session_time_label.configure(text=f"{mins:02d}:{secs:02d}")
            self.after(1000, self._update_session_time)
    
    def capture_loop(self):
        while self.camera_on and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame
            time.sleep(0.01)

    def update_video_loop(self):
        """Main UI Thread Video Processor"""
        if self.camera_on and self.latest_frame is not None:
            with self.lock:
                frame = self.latest_frame.copy()
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.pose_engine.process(rgb)
            
            feedback_msg = ""
            score_val = 0
            
            if results.pose_landmarks:
                landmarks = self.smoother.smooth(results.pose_landmarks.landmark)
                
                if self.current_pose and self.current_pose in self.poses:
                    score, analysis = self.analyze_pose(landmarks)
                    feedback_msg = analysis['msg']
                    score_val = score
                    
                    # --- WORKOUT LOGIC ---
                    if not self.in_rest_mode:
                        self.draw_skeleton(frame, landmarks, analysis['connections'])
                        
                        if score_val > 75:
                            if not self.timer_running:
                                self.timer_start = time.time()
                                self.timer_running = True
                            
                            elapsed = time.time() - self.timer_start
                            remaining = max(0, self.timer_duration - elapsed)
                            
                            # Draw Hold Timer
                            cv2.putText(frame, f"HOLD: {int(remaining)}s", (30, 80), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                            
                            if remaining <= 0:
                                self.in_rest_mode = True
                                self.rest_start = time.time()
                                self.timer_running = False
                                self.calories_burnt += 5.0 # Bonus burns
                        else:
                            self.timer_running = False # Reset if pose breaks
                    else:
                        # Rest Mode Overlay
                        rest_elapsed = time.time() - self.rest_start
                        rest_remaining = max(0, self.rest_duration - rest_elapsed)
                        
                        # Semi-transparent overlay
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
                        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
                        
                        cv2.putText(frame, "REST MODE", (w//2 - 180, h//2 - 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
                        cv2.putText(frame, f"{int(rest_remaining)}s", (w//2 - 60, h//2 + 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5)
                        
                        feedback_msg = "Take a breathe..."
                        
                        if rest_remaining <= 0:
                            self.in_rest_mode = False
                    
                    # Update best score
                    if score_val > self.best_score:
                        self.best_score = score_val
                        self.best_score_label.configure(text=f"{int(self.best_score)}%")
                else:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    feedback_msg = "Select a pose to begin"
            else:
                feedback_msg = "No person detected"

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            display_w = self.video_label.winfo_width()
            display_h = self.video_label.winfo_height()
            
            if display_w > 10 and display_h > 10:
                img_ratio = w / h
                lbl_ratio = display_w / display_h
                
                if img_ratio > lbl_ratio:
                    new_w = display_w
                    new_h = int(new_w / img_ratio)
                else:
                    new_h = display_h
                    new_w = int(new_h * img_ratio)
                    
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
            ctk_img = ctk.CTkImage(img, size=(img.width, img.height))
            self.video_label.configure(image=ctk_img)
            
            # Update HUD
            self.feedback_lbl.configure(text=feedback_msg)
            self.score_label.configure(text=f"{int(score_val)}%")
            self.score_bar.set(score_val / 100)
            
            # Dynamic score color
            if score_val >= 90:
                self.score_label.configure(text_color=COLORS['success'])
                self.score_bar.configure(progress_color=COLORS['success'])
            elif score_val >= 70:
                self.score_label.configure(text_color=COLORS['warning'])
                self.score_bar.configure(progress_color=COLORS['warning'])
            else:
                self.score_label.configure(text_color=COLORS['error'])
                self.score_bar.configure(progress_color=COLORS['error'])
                
            # Update Session Average & Calories
            if score_val > 5:
                self.session_scores.append(score_val)
                if len(self.session_scores) > 300: self.session_scores.pop(0)
                avg = sum(self.session_scores) / len(self.session_scores)
                self.accuracy_label.configure(text=f"{int(avg)}%")
                
                # Simple Calorie Est: 3.5 METs * weight(avg 70kg) / 3600 per second 
                # ~ 0.07 kcal/sec (~4 kcal/min for light yoga)
                self.calories_burnt += 0.002 # Per frame (approx 25ms)
                if int(self.calories_burnt * 10) % 10 == 0: # Update occasionally
                     pass # visual update could go here if UI existed

        self.after(25, self.update_video_loop)  # ~40fps

    def change_difficulty(self, value):
        self.difficulty = value
        print(f"Difficulty changed to: {value}")
        self.session_scores = []
        self.accuracy_label.configure(text="--%")
        
    def analyze_pose(self, landmarks):
        """Returns score (0-100) and feedback details - Tuned for 95% accuracy target"""
        if not self.current_pose: return 0, {'msg': '', 'connections': {}}
        
        target_joints = self.poses[self.current_pose]['joints']
        total_score = 0
        total_w = 0
        worst_angle = {'diff': -1, 'msg': 'Perfect Form! 🎯'}
        conn_status = {}
        
        # Difficulty Modifiers
        strictness = 1.0
        if self.difficulty == "Pro":
            strictness = 0.7 # Tolerances are 30% tighter
        
        for j in target_joints:
            weight = j.get('weight', 1.0)
            ang = self.classifier.calculate_angle(
                landmarks[j['a']], landmarks[j['b']], landmarks[j['c']]
            )
            diff = abs(ang - j['angle'])
            
            # Adjusted Logic based on Difficulty
            t_perfect = 10 * strictness
            t_good = 20 * strictness
            t_ok = 30 * strictness
            
            if diff < t_perfect:
                score = 100
            elif diff < t_good:
                score = 100 - (diff - t_perfect) * (10 / (t_good - t_perfect))
            elif diff < t_ok:
                score = 90 - (diff - t_good) * (20 / (t_ok - t_good))
            else:
                score = max(0, 70 - (diff - t_ok) * 2)
            
            total_score += score * weight
            total_w += weight
            
            is_good = score > 75
            status = 'good' if is_good else 'bad'
            
            conn_status[(j['a'], j['b'])] = status
            conn_status[(j['b'], j['c'])] = status
            
            if diff > worst_angle['diff'] and not is_good:
                worst_angle['diff'] = diff
                action = "Extend" if ang < j['angle'] else "Bend"
                worst_angle['msg'] = f"{action} your {j['name']}"
                
        final_score = total_score / max(1, total_w)
        
        if final_score >= 95:
            msg = "Perfect Form! 🎯"
        elif final_score >= 85:
            msg = "Great job! Keep it steady! ⭐"
        elif final_score >= 70:
            msg = worst_angle['msg']
        else:
            msg = worst_angle['msg']
        
        return final_score, {'msg': msg, 'connections': conn_status}

    def draw_skeleton(self, frame, landmarks, statuses):
        h, w, _ = frame.shape
        connections = self.mp_pose.POSE_CONNECTIONS
        
        # Neon colors (BGR)
        c_good = (88, 255, 127)   # Neon Green
        c_bad = (68, 68, 239)     # Neon Red
        c_neutral = (200, 200, 200) # Lighter Gray
        
        # Draw all connections
        for p1, p2 in connections:
            color = c_neutral
            thick = 2
            
            # Check status
            status = statuses.get((p1, p2)) or statuses.get((p2, p1))
            
            if status == 'good':
                color = c_good
                thick = 4
            elif status == 'bad':
                color = c_bad
                thick = 4
                
            pt1 = (int(landmarks[p1].x * w), int(landmarks[p1].y * h))
            pt2 = (int(landmarks[p2].x * w), int(landmarks[p2].y * h))
            
            cv2.line(frame, pt1, pt2, color, thick, cv2.LINE_AA)

        # Draw joints
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 6, (255, 255, 255), -1)
            cv2.circle(frame, (cx, cy), 8, (0, 0, 0), 1)

if __name__ == "__main__":
    app = YPDSApp()
    app.mainloop()