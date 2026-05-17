# 🎨 AuraPath AI: Glassmorphism Visual Theme & Dynamic UX Design System

This document outlines the visual design principles, color palettes, CSS variables, interactive micro-animations, layout systems, and responsive design guidelines that define the user experience of **AuraPath AI**.

---

## 💎 1. Glassmorphism Design System

AuraPath AI uses a **high-fidelity Glassmorphism visual theme**. It relies on semi-transparent overlay cards, subtle radial ambient glows, and clean backdrop saturation filters to create a modern, sleek interface.

### Foundational CSS Variables
These design tokens are declared globally to maintain consistency across all components:

```css
:root {
    /* Color Gradients */
    --bg-dark-radial: radial-gradient(circle at 50% 50%, #0d1224 0%, #05070f 100%);
    --bg-glass-card: rgba(16, 22, 46, 0.45);
    --bg-glass-input: rgba(8, 12, 28, 0.6);
    --bg-glass-hover: rgba(24, 35, 71, 0.65);
    
    /* Neon Accent Borders & Shadows */
    --border-glass: 1px solid rgba(255, 255, 255, 0.08);
    --border-glass-active: 1px solid rgba(0, 210, 255, 0.35);
    --shadow-neon-cyan: 0 8px 32px 0 rgba(0, 210, 255, 0.12);
    --shadow-glass-card: 0 12px 40px 0 rgba(0, 0, 0, 0.4);
    
    /* Text Primaries */
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-blue: #00d2ff;
}
```

### Backdrop Saturation Filters
The glass effect is driven by the CSS `backdrop-filter` property, which applies real-time GPU-accelerated graphic filters to the elements underneath it:
```css
.glass-panel {
    background: var(--bg-glass-card);
    backdrop-filter: blur(12px) saturate(180%);
    -webkit-backdrop-filter: blur(12px) saturate(180%);
    border: var(--border-glass);
    box-shadow: var(--shadow-glass-card);
    border-radius: 16px;
}
```
*   **`blur(12px)`**: Softens underlying DOM details, making overlaid text easy to read.
*   **`saturate(180%)`**: Amplifies background colors to keep the dark space theme looking rich and vibrant.

---

## 🎨 2. Harmonious Academic Color Palette

To keep the UI clean, cohesive, and easy to read, each of our 8 target academic disciplines is mapped to a distinct HSL color code. These colors are used for charts, progress bars, highlights, and report overlays:

| Academic Discipline | Emoji | HSL Color Code | HEX Value | UI Color Association |
| :--- | :---: | :--- | :--- | :--- |
| **Computer Science** | 💻 | `hsl(190, 100%, 50%)` | `#00d2ff` | Neon Cyan |
| **Mechanical Engineering** | 🛠️ | `hsl(30, 100%, 60%)` | `#ff9f43` | Warm Orange |
| **Electrical Engineering** | 🔌 | `hsl(160, 100%, 40%)` | `#00d28a` | Emerald Green |
| **Civil Engineering** | 🏗️ | `hsl(48, 100%, 60%)` | `#ffe066` | Bright Gold |
| **Biology** | 🧬 | `hsl(162, 73%, 46%)` | `#20c997` | Cool Teal |
| **Chemistry** | 🧪 | `hsl(268, 80%, 65%)` | `#a55eea` | Royal Purple |
| **Physics** | 🌌 | `hsl(354, 100%, 65%)` | `#ff4757` | Vibrant Coral |
| **Management** | 📊 | `hsl(355, 100%, 73%)` | `#ff7675` | Pastel Rose |

---

## 🔄 3. Dynamic Micro-Animations & Keyframes

A dynamic interface that feels responsive and alive encourages interaction. AuraPath AI achieves this using smooth transitions, hover effects, and interactive elements.

### 1. Ambient Glow Animation
An atmospheric background glow simulates a high-tech floating panel:
```css
@keyframes floatGlow {
    0% { transform: translateY(0px) scale(1); opacity: 0.2; }
    50% { transform: translateY(-10px) scale(1.05); opacity: 0.28; }
    100% { transform: translateY(0px) scale(1); opacity: 0.2; }
}
.ambient-glow {
    position: absolute;
    width: 450px;
    height: 450px;
    background: radial-gradient(circle, rgba(0, 210, 255, 0.15) 0%, rgba(0, 0, 0, 0) 70%);
    animation: floatGlow 8s ease-in-out infinite;
}
```

### 2. Spring Transitions on Interactive Elements
All option chips, text fields, and action buttons use spring transition timings instead of linear interpolation:
```css
.option-chip {
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.option-chip:hover {
    transform: translateY(-2px) scale(1.02);
    background: var(--bg-glass-hover);
    border-color: var(--accent-blue);
    box-shadow: 0 4px 20px 0 rgba(0, 210, 255, 0.15);
}
```

### 3. Smooth Chat Entry Effects
Newly appended chat cards slide and fade into view smoothly:
```css
@keyframes slideInUp {
    from { transform: translateY(12px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
}
.chat-message {
    animation: slideInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}
```

---

## 📊 4. Adaptive Step-Badge Calculation

The header progress badge dynamically scales based on the student's conversation history. The step tracker evaluates the exact length of the user responses in `chatState.history` to calculate progress, adapting to either a 5-step or 6-step path:

```javascript
function updateProgressIndicators() {
    // Count user's submitted replies in conversation history to calculate current step
    const userAnswersCount = chatState.history.filter(h => h.role === "user").length;
    let stepNum = userAnswersCount + 1;
    
    // Total steps is dynamically adjusted (6 if 12th profiling is taken, 5 if 10th/Other is taken)
    let totalSteps = 6;
    if (chatState.education_level && !chatState.education_level.includes("12")) {
        totalSteps = 5;
    }
    
    if (stepNum > totalSteps) stepNum = totalSteps;
    
    turnBadge.innerText = `Step: ${stepNum} / ${totalSteps}`;
}
```

This mathematical sync guarantees:
*   The progress bar starts at `Step: 1 / 6`.
*   If the student chooses **10th Standard**, the denominator dynamically updates to `Step: 2 / 5`, keeping the step count accurate without any hardcoded adjustments.
*   Once the student submits their final scenario response, the step count stays pinned at `6 / 6` (or `5 / 5`) while the career report is generated.

---

## 📱 5. Responsive Grid Layouts

The application is built using standard **CSS Grid** and **CSS Flexbox** frameworks. It is designed to adapt to any viewport aspect ratio, offering an optimized experience across desktop, tablet, and mobile devices:

```css
/* Dual-Panel Grid (Desktop Viewports) */
.main-wrapper {
    display: grid;
    grid-template-columns: 1.1fr 0.9fr;
    gap: 24px;
    max-width: 1400px;
    margin: 0 auto;
    height: calc(100vh - 120px);
}

/* Tablet & Mobile Breakpoints */
@media (max-width: 1024px) {
    .main-wrapper {
        grid-template-columns: 1fr;
        height: auto;
        overflow-y: visible;
    }
    .xai-panel {
        height: auto;
    }
}
```

*   **Desktop Viewport**: Displays a side-by-side grid splitting the screen into a Chat Panel ($55\%$ width) and a Live Explainable ML Panel ($45\%$ width) for easy multitasking.
*   **Mobile Viewports ($<1024\text{px}$)**: Collapses into a single-column layout where the Chat Panel flows naturally above the ML telemetry panels, making the interface easy to read and use on mobile devices.
