# ClipSearchAI Design Guidelines

## Design Approach

**Reference-Based Strategy**: Draw inspiration from Linear's clean typography and minimal aesthetics combined with YouTube's video-centric interface patterns and modern AI demo platforms (Hugging Face, Replicate).

**Core Principle**: Showcase AI analysis with clarity and visual impact while maintaining focus on video content.

---

## Typography System

**Font Families**:
- Primary: Inter (via Google Fonts) - headings, UI elements, metadata
- Monospace: JetBrains Mono - timestamps, confidence scores, technical data

**Hierarchy**:
- Page titles: text-4xl to text-5xl, font-semibold
- Section headers: text-2xl to text-3xl, font-semibold
- Component labels: text-sm, font-medium, uppercase tracking-wide
- Body/descriptions: text-base, font-normal
- Metadata (timestamps/scores): text-xs to text-sm, monospace
- Confidence percentages: text-lg, font-bold, monospace

---

## Layout System

**Spacing Primitives**: Use Tailwind units of 2, 4, 6, 8, 12, 16, 24
- Consistent component padding: p-6 or p-8
- Section spacing: space-y-8 or space-y-12
- Grid gaps: gap-4 for tight grids, gap-6 for breathing room

**Container Strategy**:
- Max-width: max-w-7xl for main content areas
- Full-width video players with contained controls
- Asymmetric layouts where analysis steps appear alongside video preview

---

## Component Library

### Navigation
- Minimal top navigation bar (h-16)
- Logo/brand left, page links center-right
- Sticky positioning for persistent access

### Frame Analysis Page

**Video Display Section**:
- Large video preview area (16:9 aspect ratio preferred)
- Positioned prominently at page top or left column
- Playback controls beneath video

**Analysis Timeline**:
- Horizontal scrollable timeline showing frame thumbnails (h-20 to h-24)
- Frame numbers/timestamps beneath each thumbnail
- Active frame highlighted with border treatment
- Smooth scroll-snap behavior

**Step-by-Step Analysis Cards**:
- Vertical stack or masonry grid of analysis steps
- Each card contains:
  - Frame thumbnail (aspect-ratio-video)
  - Timestamp badge (top-right overlay)
  - Detected objects/actions list with confidence bars
  - Scene description text
- Cards use rounded-xl borders
- Generous internal padding (p-6)

**Confidence Visualization**:
- Horizontal progress bars for confidence scores
- Percentages displayed with monospace font
- Bars transition smoothly as user scrolls through frames

### Search Page

**Search Interface**:
- Hero-style search bar centered at top (max-w-3xl)
- Large input field (h-14 to h-16) with prominent placeholder
- Search icon prefix, clear button suffix
- Example queries displayed below as clickable chips

**Results Grid**:
- 3-4 column responsive grid (grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4)
- Each result card contains:
  - Video thumbnail (aspect-ratio-video)
  - Timestamp overlay badge (bottom-left)
  - Confidence score badge (top-right)
  - Match description beneath thumbnail (2-3 lines, text-sm)
- Hover effect: subtle elevation change
- Cards use rounded-lg borders

**Empty/Loading States**:
- Skeleton loading cards matching result grid layout
- Empty state with illustration placeholder and suggested queries

### Video Player Integration
- Custom play button overlay on thumbnails
- Click behavior jumps to timestamp and auto-plays
- Minimal chrome design for embedded player
- Progress bar showing analyzed segments

---

## Animations

**Minimal, Purposeful Motion**:
- Subtle fade-in for analysis cards as they appear (duration-300)
- Smooth confidence bar fills (transition-all duration-500)
- Hover elevation on result cards (transform scale-105)
- Timeline scroll with momentum

**Avoid**: Excessive parallax, complex scroll-triggered animations, distracting background motion

---

## Images

**Frame Thumbnails**: Essential throughout - use placeholder video frames or stock footage stills representing different scenes (action, dialogue, landscapes)

**Empty State Illustrations**: Simple line art or icon representing video analysis/search

**No Hero Image**: This is a utility-focused demo app - lead directly with functionality (search bar or analysis interface)

---

## Page-Specific Layouts

### Frame Analysis Page
- Two-column desktop layout: Video player (60%) | Analysis sidebar (40%)
- Mobile: Stacked vertically with sticky video player
- Timeline spans full width beneath video
- Analysis cards scroll independently in sidebar

### Search Page
- Centered search interface with generous top spacing (pt-24)
- Results grid begins after modest gap (mt-12)
- Filters sidebar (optional): Left rail with frame rate, confidence threshold sliders
- Pagination/infinite scroll at bottom

---

## Accessibility

- Focus states visible on all interactive elements (ring-2 ring-offset-2)
- ARIA labels for video controls and confidence scores
- Keyboard navigation through timeline frames (arrow keys)
- Sufficient contrast for all text over video thumbnails
- Screen reader announcements for confidence percentages