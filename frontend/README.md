# Video Segment Retrieval Frontend

A modern, responsive frontend for natural language video segment search and retrieval.

## 🎯 Enhanced Features

### 🧠 Enhanced Search Interface (NEW)
- **AI-Powered Query Enhancement** with real-time query improvement suggestions
- **Smart Auto-complete** with context-aware suggestions (300ms debounce)
- **Enhanced Search Results** showing query analysis and confidence scores
- **Multi-Modal Search** leveraging visual, textual, and semantic features
- **Query Analytics** with enhancement statistics and performance metrics
- **Prominent search bar** with enhanced visual feedback
- **Keyboard submission** support (Enter key) with enhanced query processing
- **Loading states** with enhanced search progress indicators
- **Accessibility features** with ARIA labels and focus management

### Enhanced Results Display
- **Responsive grid layout** (1-4 columns based on screen size)
- **Enhanced thumbnail previews** with detected objects and scene information
- **Multi-factor relevance scoring** with enhanced confidence indicators
- **Rich metadata display** showing extracted features (objects, scenes, text)
- **Query enhancement feedback** showing original vs. enhanced queries
- **Hover effects** and focus states with enhanced information tooltips
- **Click-to-play** functionality with enhanced segment metadata

### Video Player Integration
- **Modern HTML5 video player** with custom controls
- **Auto-seek to segment timestamp** when clicked
- **Auto-play support** with fallback handling
- **Visual segment indicators** showing current playing segment
- **Full-screen support** and volume controls
- **Progress bar** with click-to-seek functionality

### Video Management
- **Admin interface** for video library management
- **Drag-and-drop upload** with progress tracking
- **Process videos** for indexing and search
- **Health monitoring** for backend services
- **Bulk operations** and individual video management

## 🏗️ Architecture

### Component Structure
```
components/
├── SearchBar.tsx          # Main search interface
├── SegmentList.tsx        # Grid of search results
├── SegmentItem.tsx        # Individual result card
└── VideoPlayer.tsx        # Enhanced video player
```

### Page Structure
```
app/
├── page.tsx              # Main search interface
├── manage/page.tsx       # Video management
└── layout.tsx            # Shared navigation
```

### Type Definitions
```
types/
└── index.ts              # TypeScript interfaces
```

### API Integration
```
lib/
└── api.ts                # Backend API functions
```

## 🎨 Design System

### Colors
- **Primary**: Blue (`blue-500`, `blue-600`, `blue-700`)
- **Success**: Green (`green-500`, `green-600`)
- **Error**: Red (`red-500`, `red-600`)
- **Neutral**: Gray scale (`gray-50` to `gray-900`)

### Layout
- **Responsive grid**: 1/2/3/4 columns based on breakpoint
- **Max width**: 7xl (1280px) with responsive padding
- **Spacing**: Consistent 4/6/8 unit spacing system

### Typography
- **Headings**: Font weights 600-800
- **Body**: Regular font weight with good contrast
- **Monospace**: For time displays and technical info

## 🚀 Usage

### Search Interface
1. **Enter query** in the search bar
2. **View results** in responsive grid
3. **Click segment** to start playing
4. **Toggle debug mode** to see relevance scores

### Video Management
1. **Upload videos** via drag-and-drop or file picker
2. **Process videos** to enable search indexing
3. **Monitor health** of backend services
4. **Delete videos** when no longer needed

## 🔧 Configuration

### Environment Variables
```bash
NEXT_PUBLIC_API_URL=http://localhost:8090  # Backend API URL
```

### API Endpoints
- `POST /api/search` - Search video segments
- `GET /api/videos` - List videos
- `POST /api/videos/:id/process` - Process video
- `DELETE /api/videos/:id` - Delete video
- `POST /api/videos/upload` - Upload video

## 🎯 Key Requirements Fulfilled

### ✅ Search Area
- [x] Prominent text input with placeholder
- [x] Real-time search with debouncing
- [x] Keyboard submission (Enter key)
- [x] Visual loading states

### ✅ Results Area
- [x] Grid layout of thumbnails
- [x] Video title and segment timing
- [x] Description/transcript snippets
- [x] Relevance score sorting
- [x] Hover/focus state highlights

### ✅ Video Player Integration
- [x] Single embedded player component
- [x] Auto-seek to timestamp on click
- [x] Auto-play functionality
- [x] Visual active segment indication
- [x] Modern player controls

### ✅ Technical Implementation
- [x] React + TypeScript components
- [x] Tailwind CSS styling
- [x] Custom debounce implementation
- [x] Accessibility features
- [x] Responsive design
- [x] API integration layer

## 🧪 Development

### Mock Data
The application includes mock data for development and demo purposes. Replace the mock segments in `app/page.tsx` with real API calls as needed.

### API Integration
The `lib/api.ts` file provides typed functions for all backend interactions. The search interface automatically falls back to mock data if the API is unavailable.

### Customization
- Modify colors in `tailwind.config.js`
- Adjust component props in `types/index.ts`
- Update API endpoints in `lib/api.ts`
- Customize layouts in component files

## 📱 Responsive Design

- **Mobile**: Single column, touch-friendly controls
- **Tablet**: 2-column grid, medium player size
- **Desktop**: 3-4 column grid, full-featured player
- **Large screens**: Optimized layout with max-width constraints

## ♿ Accessibility

- **Keyboard navigation** throughout the interface
- **ARIA labels** for screen readers
- **Focus management** and visible focus indicators
- **Color contrast** meeting WCAG guidelines
- **Screen reader** friendly result announcements
