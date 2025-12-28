# 🏔️ TopoBox

**Generative Topographic Box Creator**

TopoBox is a Streamlit-based application for designing and manufacturing 3D topographic landscape boxes. It fetches real-world elevation data and converts it into laser-ready vector layers, complete with structural jigs, alignment dowels, and optimized geometry.

![TopoBox Concept](https://streamlit.io/images/brand/streamlit-mark-color.png) *(Replace with actual screenshot if available)*

## 🚀 Hybrid Workflow

TopoBox is designed to work in a **Hybrid Mode** to balance accessibility with performance:

### 1. 🌐 Creator Mode (Web)
*   **Live Demo**: [https://topobox.streamlit.app/](https://topobox.streamlit.app/)
*   **Purpose**: Design, Explore, and Preview.
*   **Where**: Streamlit Cloud (Browser).
*   **Features**: Fast terrain preview, intuitive UI, "Download Project" (Save), "Resume Project" (Load).
*   **Performance**: Uses "Fast Preview" mode (skips heavy engineering calculations) for instant feedback.

### 2. 💻 Maker Mode (Local)
*   **Purpose**: Heavy Engineering, Validation, and Manufacturing Export.
*   **Where**: Local Machine (High-Performance Mac/PC).
*   **Features**: Full Boolean operations, Auto-Healing, Bridge Generation, Smart Jig creation, DXF/SVG/PDF Export.
*   **Workflow**: Load the JSON file from the Creator, let the app re-process it with full precision, and export the files for the laser cutter.

---

## 🛠️ Setup & Installation

### A. Local Installation (Maker Mode)

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/topobox.git
    cd topobox
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **API Key Setup**
    *   Get a free API Key from [OpenTopography](https://opentopography.org/).
    *   Create a file named `OpenTopography_API_key.txt` in the root folder.
    *   Paste your API key inside (just the key, no quotes).

4.  **Run the App**
    ```bash
    streamlit run app.py
    ```

### B. Cloud Deployment (Creator Mode)

1.  **Push to GitHub**: Ensure your code is on GitHub.
2.  **Deploy on Streamlit Cloud**: Connect your repo.
3.  **Configure Secrets**:
    *   Go to **App Settings** -> **Secrets**.
    *   Add the following configuration:
        ```toml
        DEPLOYMENT_MODE = "web"
        OPENTOPOGRAPHY_API_KEY = "YOUR_LONG_API_KEY_HERE"
        ```
    *   *Note: `DEPLOYMENT_MODE = "web"` enables the File Uploader/Downloader logic and Fast Preview optimizations.*

---

## 📖 Usage Guide

1.  **Select Location**: Enter Lat/Lon or browse the map.
2.  **Define Box**: Set physical dimensions (Width, Height, Depth) and Material Thickness.
3.  **Preview**: Click **Generate Preview**.
    *   *Creator Mode*: Fast, approximate visual.
    *   *Maker Mode*: Detailed structural preview.
4.  **Layer Analysis**: Use the slider to inspect individual strata.
5.  **Save/Share**:
    *   **Creator**: Click "Save Project" to download a `.json` file. Send this to the Maker.
    *   **Maker**: Click "Export" to generate the manufacturing ZIP file (DXF/PDF/SVG).

---

## ⚙️ Advanced Features

*   **Auto-Heal**: Automatically detects partial "bridges" (structural weaknesses) and fuses them.
*   **Smart Jig**: Generates a grid-based assembly jig that locks into the box frame.
*   **Optimization**: 
    *   **Resolution**: 1200px (Balanced for Cloud).
    *   **Simplification**: 0.05mm tolerance for geometry, 0.2mm for 3D visualization.

## 📄 License
[Your License Here]
