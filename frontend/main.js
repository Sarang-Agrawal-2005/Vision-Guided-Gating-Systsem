// Global variables and configuration
const API_BASE = 'http://localhost:8000';

// Zone Management System
let zoneDrawingSystem = {
    canvas: null,
    ctx: null,
    baselineImage: null,
    isDrawing: false,
    currentPoints: [],
    zones: [],
    canvasScale: 1,
    canvasOffset: { x: 0, y: 0 },
    imageLoaded: false,
    retryCount: 0,
    maxRetries: 5
};

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('🌟 DOM Content Loaded - Starting initialization...');
    initNavigation();
    initMobileMenu();
    initSmoothScrolling();
    initPageNavigation();
    addInteractiveEffects();
    
    // Initialize dashboard features with proper timing
    setTimeout(() => {
        initDashboardFeatures();
        // Start watching for baseline image changes
        watchForBaselineImage();
        // Fix for baseline image loading
        fixBaselineImageLoading();
    }, 100);
    
    console.log('🎉 All initialization complete');
});

// Initialize navigation functionality
function initNavigation() {
    const navbar = document.getElementById('navbar');
    if (navbar) {
        // Add scroll effect to navbar
        window.addEventListener('scroll', function() {
            if (window.scrollY > 50) {
                navbar.classList.add('scrolled');
            } else {
                navbar.classList.remove('scrolled');
            }
        });
    }
}

// Initialize mobile menu
function initMobileMenu() {
    const mobileMenuBtn = document.getElementById('mobileMenuBtn');
    const navLinks = document.getElementById('navLinks');
    
    if (mobileMenuBtn && navLinks) {
        mobileMenuBtn.addEventListener('click', function() {
            navLinks.classList.toggle('active');
            mobileMenuBtn.classList.toggle('active');
            
            // Animate hamburger menu
            const spans = mobileMenuBtn.querySelectorAll('span');
            if (mobileMenuBtn.classList.contains('active')) {
                spans[0].style.transform = 'rotate(45deg) translate(5px, 5px)';
                spans[1].style.opacity = '0';
                spans[2].style.transform = 'rotate(-45deg) translate(7px, -6px)';
            } else {
                spans[0].style.transform = 'none';
                spans[1].style.opacity = '1';
                spans[2].style.transform = 'none';
            }
        });

        // Close mobile menu when clicking on a link
        const navItems = navLinks.querySelectorAll('a');
        navItems.forEach(item => {
            item.addEventListener('click', function() {
                navLinks.classList.remove('active');
                mobileMenuBtn.classList.remove('active');
                
                // Reset hamburger menu
                const spans = mobileMenuBtn.querySelectorAll('span');
                spans[0].style.transform = 'none';
                spans[1].style.opacity = '1';
                spans[2].style.transform = 'none';
            });
        });

        // Close mobile menu when clicking outside
        document.addEventListener('click', function(event) {
            if (!mobileMenuBtn.contains(event.target) && !navLinks.contains(event.target)) {
                navLinks.classList.remove('active');
                mobileMenuBtn.classList.remove('active');
                
                // Reset hamburger menu
                const spans = mobileMenuBtn.querySelectorAll('span');
                spans[0].style.transform = 'none';
                spans[1].style.opacity = '1';
                spans[2].style.transform = 'none';
            }
        });
    }
}

// Initialize smooth scrolling for anchor links
function initSmoothScrolling() {
    const links = document.querySelectorAll('a[href^="#"]');
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            const href = link.getAttribute('href');
            // Skip if it's just a hash
            if (href === '#') return;
            
            e.preventDefault();
            const target = document.querySelector(href);
            if (target) {
                const offsetTop = target.offsetTop - 80; // Account for fixed navbar
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
            }
        });
    });
}

// Initialize page navigation buttons
function initPageNavigation() {
    // Get all navigation buttons
    const getStartedBtn = document.getElementById('getStartedBtn');
    const startBuildingBtn = document.getElementById('startBuildingBtn');
    const patentBtn = document.getElementById('patentBtn');
    const backToHomeBtn = document.getElementById('backToHomeBtn');

    // Navigation to dashboard page
    if (getStartedBtn) {
        getStartedBtn.addEventListener('click', function(e) {
            e.preventDefault();
            navigateToPipeline();
        });
    }

    if (startBuildingBtn) {
        startBuildingBtn.addEventListener('click', function(e) {
            e.preventDefault();
            navigateToPipeline();
        });
    }

    // Navigation to patent page
    if (patentBtn) {
        patentBtn.addEventListener('click', function(e) {
            e.preventDefault();
            navigateToPatent();
        });
    }

    // Navigation back to home
    if (backToHomeBtn) {
        backToHomeBtn.addEventListener('click', function(e) {
            e.preventDefault();
            navigateToHome();
        });
    }
}

// Navigation functions
function navigateToPipeline() {
    showLoadingState();
    setTimeout(() => {
        window.location.href = './dashboard.html';
    }, 500);
}

function navigateToPatent() {
    showLoadingState();
    setTimeout(() => {
        window.location.href = './patent.html';
    }, 500);
}

function navigateToHome() {
    showLoadingState();
    setTimeout(() => {
        window.location.href = './index.html';
    }, 500);
}

// Show loading state
function showLoadingState() {
    const body = document.body;
    
    // Create loading overlay
    const loadingOverlay = document.createElement('div');
    loadingOverlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(255, 255, 255, 0.9);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 9999;
        backdrop-filter: blur(5px);
    `;

    // Create loading spinner
    const spinner = document.createElement('div');
    spinner.style.cssText = `
        width: 40px;
        height: 40px;
        border: 4px solid #e2e8f0;
        border-top: 4px solid #0ea5e9;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    `;

    // Add CSS animation
    const style = document.createElement('style');
    style.textContent = `
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    `;
    document.head.appendChild(style);

    loadingOverlay.appendChild(spinner);
    body.appendChild(loadingOverlay);
}

// Add interactive effects to buttons
function addInteractiveEffects() {
    const buttons = document.querySelectorAll('.btn');
    buttons.forEach(button => {
        button.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px)';
        });

        button.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });

        button.addEventListener('mousedown', function() {
            this.style.transform = 'translateY(0) scale(0.98)';
        });

        button.addEventListener('mouseup', function() {
            this.style.transform = 'translateY(-2px) scale(1)';
        });
    });
}

// Format duration helper function (moved to global scope)
function formatDuration(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}

// Enhanced Dashboard Features
function initDashboardFeatures() {
    console.log('🚀 Initializing enhanced dashboard features...');
    
    // Initialize dashboard tabs
    initDashboardTabs();
    
    // Initialize enhanced video upload functionality
    initEnhancedVideoUpload();
    
    console.log('✅ Enhanced dashboard features initialized');
}

// Enhanced tab switching with proper zone initialization
function initDashboardTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanels = document.querySelectorAll('.tab-panel');

    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const targetTab = this.getAttribute('data-tab');
            
            // Remove active class from all buttons and panels
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabPanels.forEach(panel => panel.classList.remove('active'));
            
            // Add active class to clicked button and corresponding panel
            this.classList.add('active');
            const targetPanel = document.getElementById(targetTab);
            if (targetPanel) {
                targetPanel.classList.add('active');
            }
            
            // Special handling for analyze (zones) tab
            if (targetTab === 'analyze') {
                console.log('🎯 Switching to zones tab...');
                // Reset retry count when switching to zones tab
                zoneDrawingSystem.retryCount = 0;
                // Small delay to ensure DOM is ready
                setTimeout(() => {
                    initZoneDrawing();
                }, 100);
            }
        });
    });
}

// Fix for baseline image loading in zones tab
function fixBaselineImageLoading() {
    // Watch for tab switches to analyze (zones) tab
    const analyzeTab = document.querySelector('[data-tab="analyze"]');
    if (analyzeTab) {
        analyzeTab.addEventListener('click', function() {
            setTimeout(() => {
                // Force reload baseline image to canvas
                const baselineFrame = document.getElementById('baselineFrame');
                const canvas = document.getElementById('zoneCanvas');
                
                if (baselineFrame && baselineFrame.src && canvas) {
                    const ctx = canvas.getContext('2d');
                    const img = new Image();
                    
                    img.onload = function() {
                        // Store image in zone drawing system
                        zoneDrawingSystem.baselineImage = img;
                        zoneDrawingSystem.imageLoaded = true;
                        
                        // Resize canvas to image
                        const maxWidth = 800;
                        const maxHeight = 600;
                        let { width, height } = img;
                        
                        const scaleX = maxWidth / width;
                        const scaleY = maxHeight / height;
                        const scale = Math.min(scaleX, scaleY, 1);
                        
                        width *= scale;
                        height *= scale;
                        
                        canvas.width = width;
                        canvas.height = height;
                        zoneDrawingSystem.canvasScale = scale;
                        
                        // Draw image to canvas
                        ctx.drawImage(img, 0, 0, width, height);
                        
                        // Hide overlay
                        const overlay = document.getElementById('canvasOverlay');
                        if (overlay) {
                            overlay.classList.add('hidden');
                        }
                        
                        console.log('✅ Baseline image loaded to canvas');
                    };
                    
                    img.src = baselineFrame.src;
                }
            }, 200);
        });
    }
}

// Enhanced Video Upload Functionality
function initEnhancedVideoUpload() {
    const uploadZone = document.getElementById('uploadZone');
    const videoInput = document.getElementById('videoInput');
    const uploadProgress = document.getElementById('uploadProgress');
    const videoPreview = document.getElementById('videoPreview');
    const progressFill = document.getElementById('progressFill');
    const progressPercentage = document.getElementById('progressPercentage');
    const progressStatus = document.getElementById('progressStatus');

    if (!uploadZone || !videoInput) return;

    // Click to upload
    uploadZone.addEventListener('click', () => {
        if (!uploadZone.classList.contains('uploading')) {
            videoInput.click();
        }
    });

    // Enhanced drag and drop functionality
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        if (!uploadZone.contains(e.relatedTarget)) {
            uploadZone.classList.remove('dragover');
        }
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleVideoUpload(files[0]);
        }
    });

    // File input change
    videoInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleVideoUpload(e.target.files[0]);
        }
    });

    // Upload new video button
    const uploadNewBtn = document.getElementById('uploadNewBtn');
    if (uploadNewBtn) {
        uploadNewBtn.addEventListener('click', () => {
            resetUploadState();
            videoInput.click();
        });
    }

    // Confirm baseline button
    const confirmBaselineBtn = document.getElementById('confirmBaselineBtn');
    if (confirmBaselineBtn) {
        confirmBaselineBtn.addEventListener('click', () => {
            showNotification('Baseline confirmed successfully! Ready for analysis.', 'success');
            // Force reload baseline image to canvas if on zones tab
            setTimeout(() => {
                const zonesTab = document.getElementById('analyze');
                if (zonesTab && zonesTab.classList.contains('active')) {
                    forceLoadBaselineImageWithRetry();
                }
            }, 500);
        });
    }

    // Handle video upload with enhanced features
    async function handleVideoUpload(file) {
        // Validate file type
        if (!file.type.startsWith('video/')) {
            showNotification('Please select a valid video file', 'error');
            return;
        }

        // Check file size (100MB limit)
        const maxSize = 100 * 1024 * 1024; // 100MB in bytes
        if (file.size > maxSize) {
            showNotification('File size exceeds 100MB limit', 'error');
            return;
        }

        // Show upload state
        showUploadProgress();
        
        try {
            // Create FormData
            const formData = new FormData();
            formData.append('file', file);

            // Upload with progress tracking
            const response = await uploadWithProgress(formData, (progress) => {
                updateProgress(progress);
            });

            if (response.ok) {
                const result = await response.json();
                if (progressStatus) {
                    progressStatus.textContent = 'Processing video...';
                }
                
                // Get video info and first frame
                await displayVideoPreview(result.video_id, file.name);
                hideUploadProgress();
                showNotification('Video uploaded successfully!', 'success');
            } else {
                throw new Error('Upload failed');
            }

        } catch (error) {
            console.error('Upload error:', error);
            hideUploadProgress();
            showNotification('Upload failed. Please try again.', 'error');
        }
    }

    // Upload with progress tracking
    function uploadWithProgress(formData, onProgress) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();

            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const progress = (e.loaded / e.total) * 100;
                    onProgress(progress);
                }
            });

            xhr.addEventListener('load', () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    resolve({
                        ok: true,
                        json: () => Promise.resolve(JSON.parse(xhr.responseText))
                    });
                } else {
                    reject(new Error(`HTTP ${xhr.status}: ${xhr.statusText}`));
                }
            });

            xhr.addEventListener('error', () => {
                reject(new Error('Network error'));
            });

            xhr.open('POST', `${API_BASE}/api/video/upload`);
            xhr.send(formData);
        });
    }

    // UI Helper Functions
    function showUploadProgress() {
        uploadZone.classList.add('uploading');
        if (uploadProgress) uploadProgress.classList.remove('hidden');
        if (videoPreview) videoPreview.classList.add('hidden');
        updateProgress(0);
    }

    function hideUploadProgress() {
        uploadZone.classList.remove('uploading');
        if (uploadProgress) uploadProgress.classList.add('hidden');
    }

    function updateProgress(progress) {
        if (progressFill) progressFill.style.width = `${progress}%`;
        if (progressPercentage) progressPercentage.textContent = `${Math.round(progress)}%`;
    }

    function resetUploadState() {
        hideUploadProgress();
        if (videoPreview) videoPreview.classList.add('hidden');
        videoInput.value = '';
    }
}

// Enhanced video preview display function with guaranteed canvas loading
async function displayVideoPreview(videoId, fileName) {
    console.log('🎬 Displaying video preview for:', videoId, fileName);
    
    try {
        // Get video metadata
        console.log('📊 Fetching metadata...');
        const metadataResponse = await fetch(`${API_BASE}/api/video/${videoId}`);
        
        if (!metadataResponse.ok) {
            throw new Error(`Metadata fetch failed: ${metadataResponse.status}`);
        }
        
        const videoInfo = await metadataResponse.json();
        console.log('📊 Video info:', videoInfo);

        // Get first frame
        console.log('🖼️ Fetching first frame...');
        const frameResponse = await fetch(`${API_BASE}/api/video/${videoId}/first-frame`);
        
        if (!frameResponse.ok) {
            throw new Error(`Frame fetch failed: ${frameResponse.status}`);
        }
        
        const frameBlob = await frameResponse.blob();
        const frameUrl = URL.createObjectURL(frameBlob);
        console.log('🖼️ Frame URL created:', frameUrl);

        // Update UI elements
        const baselineFrame = document.getElementById('baselineFrame');
        const videoName = document.getElementById('videoName');
        const videoDuration = document.getElementById('videoDuration');
        const videoResolution = document.getElementById('videoResolution');
        const videoFps = document.getElementById('videoFps');

        console.log('🎯 UI Elements found:', {
            baselineFrame: !!baselineFrame,
            videoName: !!videoName,
            videoDuration: !!videoDuration,
            videoResolution: !!videoResolution,
            videoFps: !!videoFps
        });

        if (baselineFrame) {
            // Store the blob URL globally for access
            window.currentBaselineUrl = frameUrl;
            
            baselineFrame.src = frameUrl;
            baselineFrame.onload = () => {
                console.log('✅ Baseline image loaded successfully');
                
                // Immediately try to load to canvas if on zones tab
                const zonesTab = document.getElementById('analyze');
                if (zonesTab && zonesTab.classList.contains('active')) {
                    console.log('🎯 Already on zones tab, loading image to canvas...');
                    setTimeout(() => {
                        forceLoadBaselineImageWithRetry();
                    }, 100);
                }
            };
            baselineFrame.onerror = (e) => console.error('❌ Image failed to load:', e);
        }
        
        if (videoName) videoName.textContent = fileName;
        if (videoDuration) videoDuration.textContent = videoInfo.duration_formatted || formatDuration(videoInfo.frame_count / videoInfo.fps);
        if (videoResolution) videoResolution.textContent = `${videoInfo.width} × ${videoInfo.height}`;
        if (videoFps) videoFps.textContent = `${Math.round(videoInfo.fps)} fps`;

        // Show video preview
        const videoPreview = document.getElementById('videoPreview');
        if (videoPreview) {
            videoPreview.classList.remove('hidden');
            console.log('✅ Video preview shown');
        } else {
            console.error('❌ Video preview element not found');
        }

    } catch (error) {
        console.error('❌ Error displaying video preview:', error);
        showNotification('Error loading video preview: ' + error.message, 'error');
    }
}

// Initialize zone drawing functionality with guaranteed baseline loading
function initZoneDrawing() {
    console.log('🎨 Initializing zone drawing system...');
    const canvas = document.getElementById('zoneCanvas');
    const zoneConfigForm = document.getElementById('zoneConfigForm');
    
    if (!canvas) {
        console.log('❌ Zone canvas not found');
        return;
    }

    // Always show the zone configuration form
    if (zoneConfigForm) {
        zoneConfigForm.classList.remove('hidden');
        console.log('✅ Zone configuration form is now visible');
    }

    zoneDrawingSystem.canvas = canvas;
    zoneDrawingSystem.ctx = canvas.getContext('2d');
    
    // Load zones from storage
    loadZonesFromStorage();
    
    // Setup event listeners
    setupZoneDrawingEvents();
    
    // Force load baseline image
    setTimeout(() => {
        forceLoadBaselineImageWithRetry();
    }, 200);
    
    console.log('✅ Zone drawing system initialized');
}

// Enhanced baseline image loading with guaranteed retry mechanism
async function forceLoadBaselineImageWithRetry() {
    console.log(`🖼️ Force loading baseline image (attempt ${zoneDrawingSystem.retryCount + 1}/${zoneDrawingSystem.maxRetries})...`);
    
    // Try multiple sources for the baseline image
    let imageSrc = null;
    
    // Method 1: Check global stored URL
    if (window.currentBaselineUrl) {
        imageSrc = window.currentBaselineUrl;
        console.log('🔍 Using stored baseline URL:', imageSrc);
    }
    
    // Method 2: Check baseline frame element
    if (!imageSrc) {
        const baselineFrame = document.getElementById('baselineFrame');
        if (baselineFrame && baselineFrame.src && baselineFrame.src !== window.location.href) {
            imageSrc = baselineFrame.src;
            console.log('🔍 Using baseline frame src:', imageSrc);
        }
    }
    
    if (!imageSrc) {
        console.log('❌ No baseline image source available');
        if (zoneDrawingSystem.retryCount < zoneDrawingSystem.maxRetries) {
            zoneDrawingSystem.retryCount++;
            setTimeout(() => {
                forceLoadBaselineImageWithRetry();
            }, 1000);
            return;
        }
        showImageLoadingError();
        return;
    }
    
    try {
        await loadImageToCanvas(imageSrc);
        zoneDrawingSystem.retryCount = 0; // Reset on success
    } catch (error) {
        console.error('❌ Failed to load baseline image:', error);
        
        if (zoneDrawingSystem.retryCount < zoneDrawingSystem.maxRetries) {
            zoneDrawingSystem.retryCount++;
            setTimeout(() => {
                forceLoadBaselineImageWithRetry();
            }, 1000);
        } else {
            showImageLoadingError();
        }
    }
}

// Enhanced image loading function with proper error handling
function loadImageToCanvas(imageSrc) {
    return new Promise((resolve, reject) => {
        console.log('📥 Loading image to canvas:', imageSrc);
        
        const img = new Image();
        
        // Don't set crossOrigin for blob URLs
        if (!imageSrc.startsWith('blob:')) {
            img.crossOrigin = 'anonymous';
        }
        
        img.onload = () => {
            console.log('✅ Image loaded successfully, drawing to canvas...');
            console.log('📐 Image dimensions:', img.width, 'x', img.height);
            
            try {
                zoneDrawingSystem.baselineImage = img;
                zoneDrawingSystem.imageLoaded = true;
                
                resizeCanvasToImage(img);
                redrawCanvas();
                hideImageLoadingError();
                
                console.log('🎨 Canvas updated with baseline image');
                resolve(img);
            } catch (canvasError) {
                console.error('❌ Error drawing to canvas:', canvasError);
                reject(canvasError);
            }
        };
        
        img.onerror = (error) => {
            console.error('❌ Failed to load image to canvas:', error);
            reject(error);
        };
        
        // Set the source - this triggers the loading
        img.src = imageSrc;
    });
}

// Show image loading error
function showImageLoadingError() {
    const canvasOverlay = document.getElementById('canvasOverlay');
    if (canvasOverlay) {
        canvasOverlay.classList.remove('hidden');
        canvasOverlay.innerHTML = `
            <div class="drawing-instructions">
                <i class="fas fa-exclamation-triangle" style="color: #f59e0b;"></i>
                <h3 style="color: white; margin: 1rem 0;">No Baseline Image Available</h3>
                <p style="color: white;">Please upload a video and confirm baseline first</p>
                <button class="btn btn-primary" onclick="goToUploadTab()" style="margin-top: 1rem;">
                    <i class="fas fa-upload"></i>
                    Go to Upload Tab
                </button>
            </div>
        `;
    }
}

// Hide image loading error
function hideImageLoadingError() {
    const canvasOverlay = document.getElementById('canvasOverlay');
    if (canvasOverlay) {
        canvasOverlay.classList.add('hidden');
        canvasOverlay.innerHTML = `
            <div class="drawing-instructions">
                <i class="fas fa-mouse-pointer"></i>
                <p>Click on the image to add zone boundary points</p>
                <p><strong>Tip:</strong> You need at least 3 points to create a zone</p>
            </div>
        `;
    }
}

// Function to go to upload tab
function goToUploadTab() {
    const uploadTabBtn = document.querySelector('[data-tab="upload"]');
    if (uploadTabBtn) {
        uploadTabBtn.click();
    }
}

// Enhanced baseline image loading that watches for changes
function watchForBaselineImage() {
    const baselineFrame = document.getElementById('baselineFrame');
    
    if (!baselineFrame) return;
    
    // Watch for src changes
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.type === 'attributes' && mutation.attributeName === 'src') {
                console.log('🔄 Baseline image source changed, reloading...');
                // Store the new URL globally
                window.currentBaselineUrl = baselineFrame.src;
                
                setTimeout(() => {
                    // Only reload if we're on the zones tab
                    const zonesTab = document.getElementById('analyze');
                    if (zonesTab && zonesTab.classList.contains('active')) {
                        zoneDrawingSystem.retryCount = 0; // Reset retry count
                        forceLoadBaselineImageWithRetry();
                    }
                }, 300);
            }
        });
    });
    
    observer.observe(baselineFrame, {
        attributes: true,
        attributeFilter: ['src']
    });
    
    // Also watch for load events
    baselineFrame.addEventListener('load', () => {
        console.log('🔄 Baseline image loaded event detected');
        // Store the URL globally
        window.currentBaselineUrl = baselineFrame.src;
        
        setTimeout(() => {
            // Only reload if we're on the zones tab
            const zonesTab = document.getElementById('analyze');
            if (zonesTab && zonesTab.classList.contains('active')) {
                console.log('🎯 On zones tab, loading to canvas...');
                zoneDrawingSystem.retryCount = 0; // Reset retry count
                forceLoadBaselineImageWithRetry();
            }
        }, 100);
    });
}

// Setup all event listeners for zone drawing
function setupZoneDrawingEvents() {
    const startDrawingBtn = document.getElementById('startDrawingBtn');
    const clearPointsBtn = document.getElementById('clearPointsBtn');
    const undoPointBtn = document.getElementById('undoPointBtn');
    const zoneForm = document.getElementById('zoneForm');
    const cancelZoneBtn = document.getElementById('cancelZoneBtn');
    const exportZonesBtn = document.getElementById('exportZonesBtn');
    const importZonesBtn = document.getElementById('importZonesBtn');
    const importZonesInput = document.getElementById('importZonesInput');
    
    // Drawing controls
    if (startDrawingBtn) {
        startDrawingBtn.addEventListener('click', function() {
            console.log('🎯 Start drawing button clicked');
            startDrawingZone();
        });
    }
    
    if (clearPointsBtn) {
        clearPointsBtn.addEventListener('click', clearCurrentPoints);
    }
    
    if (undoPointBtn) {
        undoPointBtn.addEventListener('click', undoLastPoint);
    }
    
    // Canvas click events
    if (zoneDrawingSystem.canvas) {
        zoneDrawingSystem.canvas.addEventListener('click', handleCanvasClick);
        zoneDrawingSystem.canvas.addEventListener('mousemove', handleCanvasMouseMove);
        zoneDrawingSystem.canvas.addEventListener('dblclick', handleCanvasDoubleClick);
    }
    
    // Form events
    if (zoneForm) {
        zoneForm.addEventListener('submit', handleZoneFormSubmit);
    }
    
    if (cancelZoneBtn) {
        cancelZoneBtn.addEventListener('click', cancelZoneCreation);
    }
    
    // Import/Export events
    if (exportZonesBtn) {
        exportZonesBtn.addEventListener('click', exportZonesConfig);
    }
    
    if (importZonesBtn) {
        importZonesBtn.addEventListener('click', () => importZonesInput.click());
    }
    
    if (importZonesInput) {
        importZonesInput.addEventListener('change', importZonesConfig);
    }
    
    // Range input updates
    setupRangeInputs();
}

// Setup range input value updates
function setupRangeInputs() {
    const ranges = [
        { input: 'motionThreshold', display: 'thresholdValue' },
        { input: 'minArea', display: 'areaValue' },
        { input: 'motionFrames', display: 'framesValue' }
    ];
    
    ranges.forEach(({ input, display }) => {
        const inputEl = document.getElementById(input);
        const displayEl = document.getElementById(display);
        
        if (inputEl && displayEl) {
            inputEl.addEventListener('input', () => {
                displayEl.textContent = inputEl.value;
            });
        }
    });
}

// Resize canvas to fit image
function resizeCanvasToImage(img) {
    const maxWidth = 800;
    const maxHeight = 600;
    
    let { width, height } = img;
    
    // Calculate scale to fit within max dimensions
    const scaleX = maxWidth / width;
    const scaleY = maxHeight / height;
    const scale = Math.min(scaleX, scaleY, 1);
    
    width *= scale;
    height *= scale;
    
    zoneDrawingSystem.canvas.width = width;
    zoneDrawingSystem.canvas.height = height;
    zoneDrawingSystem.canvasScale = scale;
    
    console.log('📐 Canvas resized to:', width, 'x', height, 'scale:', scale);
    
    // Center canvas
    const container = zoneDrawingSystem.canvas.parentElement;
    const containerRect = container.getBoundingClientRect();
    zoneDrawingSystem.canvasOffset = {
        x: (containerRect.width - width) / 2,
        y: (containerRect.height - height) / 2
    };
}

// Enhanced start drawing zone function
function startDrawingZone() {
    console.log('🎨 Starting zone drawing...');
    
    if (!zoneDrawingSystem.baselineImage || !zoneDrawingSystem.imageLoaded) {
        showNotification('Please upload a video and confirm baseline first', 'error');
        // Try to reload the image
        forceLoadBaselineImageWithRetry();
        return;
    }
    
    zoneDrawingSystem.isDrawing = true;
    zoneDrawingSystem.currentPoints = [];
    
    // Update UI
    const startBtn = document.getElementById('startDrawingBtn');
    const clearBtn = document.getElementById('clearPointsBtn');
    const undoBtn = document.getElementById('undoPointBtn');
    const statusEl = document.getElementById('drawingStatus');
    const countEl = document.getElementById('pointsCount');
    
    if (startBtn) startBtn.disabled = true;
    if (clearBtn) clearBtn.disabled = false;
    if (undoBtn) undoBtn.disabled = false;
    if (statusEl) {
        statusEl.textContent = 'Click on image to add points (double-click to finish)';
        statusEl.classList.add('active');
    }
    if (countEl) countEl.textContent = 'Points: 0';
    
    zoneDrawingSystem.canvas.classList.add('drawing');
    console.log('✅ Zone drawing mode activated');
}

// Handle canvas click for point addition
function handleCanvasClick(event) {
    if (!zoneDrawingSystem.isDrawing) return;
    
    const rect = zoneDrawingSystem.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    // Convert to image coordinates
    const imageX = x / zoneDrawingSystem.canvasScale;
    const imageY = y / zoneDrawingSystem.canvasScale;
    
    zoneDrawingSystem.currentPoints.push({ x: imageX, y: imageY });
    
    // Update UI
    const countEl = document.getElementById('pointsCount');
    const statusEl = document.getElementById('drawingStatus');
    
    if (countEl) countEl.textContent = `Points: ${zoneDrawingSystem.currentPoints.length}`;
    
    if (zoneDrawingSystem.currentPoints.length >= 3) {
        if (statusEl) statusEl.textContent = 'Ready to create zone! Double-click to finish or continue adding points.';
    }
    
    redrawCanvas();
}

// Handle canvas double-click to finish zone
function handleCanvasDoubleClick(event) {
    if (!zoneDrawingSystem.isDrawing || zoneDrawingSystem.currentPoints.length < 3) return;
    
    // Focus on zone name input since form is already visible
    const zoneNameInput = document.getElementById('zoneName');
    if (zoneNameInput) {
        zoneNameInput.focus();
    }
}

// Handle canvas mouse move for preview
function handleCanvasMouseMove(event) {
    if (!zoneDrawingSystem.isDrawing || zoneDrawingSystem.currentPoints.length === 0) return;
    
    const rect = zoneDrawingSystem.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    redrawCanvas();
    
    // Draw preview line
    const ctx = zoneDrawingSystem.ctx;
    const lastPoint = zoneDrawingSystem.currentPoints[zoneDrawingSystem.currentPoints.length - 1];
    
    ctx.strokeStyle = '#FFFF00';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(lastPoint.x * zoneDrawingSystem.canvasScale, lastPoint.y * zoneDrawingSystem.canvasScale);
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.setLineDash([]);
}

// Clear current points
function clearCurrentPoints() {
    zoneDrawingSystem.currentPoints = [];
    const countEl = document.getElementById('pointsCount');
    const statusEl = document.getElementById('drawingStatus');
    
    if (countEl) countEl.textContent = 'Points: 0';
    if (statusEl) statusEl.textContent = 'Click on image to add points (double-click to finish)';
    
    redrawCanvas();
}

// Undo last point
function undoLastPoint() {
    if (zoneDrawingSystem.currentPoints.length > 0) {
        zoneDrawingSystem.currentPoints.pop();
        const countEl = document.getElementById('pointsCount');
        const statusEl = document.getElementById('drawingStatus');
        
        if (countEl) countEl.textContent = `Points: ${zoneDrawingSystem.currentPoints.length}`;
        
        if (zoneDrawingSystem.currentPoints.length === 0) {
            if (statusEl) statusEl.textContent = 'Click on image to add points (double-click to finish)';
        }
        
        redrawCanvas();
    }
}

// Redraw entire canvas
function redrawCanvas() {
    const ctx = zoneDrawingSystem.ctx;
    const canvas = zoneDrawingSystem.canvas;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw baseline image
    if (zoneDrawingSystem.baselineImage) {
        ctx.drawImage(
            zoneDrawingSystem.baselineImage,
            0, 0,
            canvas.width, canvas.height
        );
    }
    
    // Draw existing zones
    drawExistingZones();
    
    // Draw current points
    drawCurrentPoints();
}

// Draw existing zones (similar to Streamlit implementation)
function drawExistingZones() {
    const ctx = zoneDrawingSystem.ctx;
    
    zoneDrawingSystem.zones.forEach((zone, index) => {
        if (zone.coordinates.length < 3) return;
        
        const color = zone.color || '#FF0000';
        
        // Draw filled polygon with transparency
        ctx.fillStyle = color + '40'; // Add alpha
        ctx.strokeStyle = color;
        ctx.lineWidth = 4;
        
        ctx.beginPath();
        const firstPoint = zone.coordinates[0];
        ctx.moveTo(firstPoint.x * zoneDrawingSystem.canvasScale, firstPoint.y * zoneDrawingSystem.canvasScale);
        
        for (let i = 1; i < zone.coordinates.length; i++) {
            const point = zone.coordinates[i];
            ctx.lineTo(point.x * zoneDrawingSystem.canvasScale, point.y * zoneDrawingSystem.canvasScale);
        }
        
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        
        // Draw zone label with background (similar to Streamlit)
        const centerX = zone.coordinates.reduce((sum, p) => sum + p.x, 0) / zone.coordinates.length * zoneDrawingSystem.canvasScale;
        const centerY = zone.coordinates.reduce((sum, p) => sum + p.y, 0) / zone.coordinates.length * zoneDrawingSystem.canvasScale;
        
        const label = `${zone.name} (P${zone.priority})`;
        
        // Draw text background
        ctx.font = 'bold 14px Arial';
        const textMetrics = ctx.measureText(label);
        const textWidth = textMetrics.width;
        const textHeight = 16;
        
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(centerX - textWidth/2 - 5, centerY - textHeight/2 - 5, textWidth + 10, textHeight + 10);
        
        // Draw text
        ctx.fillStyle = '#FFFFFF';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, centerX, centerY);
    });
}

// Draw current points being drawn (similar to Streamlit)
function drawCurrentPoints() {
    const ctx = zoneDrawingSystem.ctx;
    
    if (zoneDrawingSystem.currentPoints.length === 0) return;
    
    // Draw points
    ctx.fillStyle = '#FFFF00';
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 2;
    
    zoneDrawingSystem.currentPoints.forEach((point, index) => {
        const x = point.x * zoneDrawingSystem.canvasScale;
        const y = point.y * zoneDrawingSystem.canvasScale;
        
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
        
        // Draw point number
        ctx.fillStyle = '#000000';
        ctx.font = 'bold 12px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText((index + 1).toString(), x, y - 20);
        ctx.fillStyle = '#FFFF00';
    });
    
    // Draw lines between points
    if (zoneDrawingSystem.currentPoints.length > 1) {
        ctx.strokeStyle = '#FFFF00';
        ctx.lineWidth = 4;
        ctx.beginPath();
        
        const firstPoint = zoneDrawingSystem.currentPoints[0];
        ctx.moveTo(firstPoint.x * zoneDrawingSystem.canvasScale, firstPoint.y * zoneDrawingSystem.canvasScale);
        
        for (let i = 1; i < zoneDrawingSystem.currentPoints.length; i++) {
            const point = zoneDrawingSystem.currentPoints[i];
            ctx.lineTo(point.x * zoneDrawingSystem.canvasScale, point.y * zoneDrawingSystem.canvasScale);
        }
        
        ctx.stroke();
        
        // Draw preview closing line if 3+ points
        if (zoneDrawingSystem.currentPoints.length >= 3) {
            ctx.setLineDash([5, 5]);
            ctx.lineWidth = 2;
            ctx.beginPath();
            const lastPoint = zoneDrawingSystem.currentPoints[zoneDrawingSystem.currentPoints.length - 1];
            ctx.moveTo(lastPoint.x * zoneDrawingSystem.canvasScale, lastPoint.y * zoneDrawingSystem.canvasScale);
            ctx.lineTo(firstPoint.x * zoneDrawingSystem.canvasScale, firstPoint.y * zoneDrawingSystem.canvasScale);
            ctx.stroke();
            ctx.setLineDash([]);
        }
    }
}

/// Handle zone form submission
function handleZoneFormSubmit(event) {
    event.preventDefault();
    
    if (zoneDrawingSystem.currentPoints.length < 3) {
        showNotification('Please add at least 3 points to create a zone', 'error');
        return;
    }
    
    const zoneName = document.getElementById('zoneName').value.trim();
    
    // Check for duplicate zone names
    if (zoneDrawingSystem.zones.some(zone => zone.name === zoneName)) {
        showNotification('Zone name already exists', 'error');
        return;
    }
    
    const newZone = {
        id: Date.now().toString(),
        name: zoneName,
        coordinates: [...zoneDrawingSystem.currentPoints],
        priority: parseInt(document.getElementById('zonePriority').value),
        threshold: parseInt(document.getElementById('motionThreshold').value),
        min_area: parseInt(document.getElementById('minArea').value),
        motion_frames: parseInt(document.getElementById('motionFrames').value),
        color: document.getElementById('zoneColor').value,
        created_at: new Date().toISOString()
    };
    
    // Save zone
    saveZone(newZone);
    
    // CRITICAL FIX: Ensure form stays visible after submission
    setTimeout(() => {
        const zoneConfigForm = document.getElementById('zoneConfigForm');
        if (zoneConfigForm) {
            zoneConfigForm.classList.remove('hidden');
            zoneConfigForm.style.display = 'block';
            zoneConfigForm.style.visibility = 'visible';
        }
    }, 100);
}


// Save zone to system
async function saveZone(zone) {
    try {
        // Save to backend
        const response = await fetch(`${API_BASE}/api/zones`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(zone)
        });
        
        if (!response.ok) {
            throw new Error('Failed to save zone to server');
        }
        
        // Add to local zones
        zoneDrawingSystem.zones.push(zone);
        
        // Save to local storage
        saveZonesToStorage();
        
        // Update UI
        renderZonesList();
        finishZoneCreation();
        
        // CRITICAL FIX: Ensure form stays visible after zone creation
        const zoneConfigForm = document.getElementById('zoneConfigForm');
        if (zoneConfigForm) {
            zoneConfigForm.classList.remove('hidden');
            zoneConfigForm.style.display = 'block';
            zoneConfigForm.style.visibility = 'visible';
        }
        
        showNotification(`Zone "${zone.name}" created successfully!`, 'success');
        
    } catch (error) {
        console.error('Error saving zone:', error);
        showNotification('Failed to save zone. Please try again.', 'error');
    }
}


// Finish zone creation
function finishZoneCreation() {
    zoneDrawingSystem.isDrawing = false;
    zoneDrawingSystem.currentPoints = [];
    
    // Reset UI
    const startBtn = document.getElementById('startDrawingBtn');
    const clearBtn = document.getElementById('clearPointsBtn');
    const undoBtn = document.getElementById('undoPointBtn');
    const statusEl = document.getElementById('drawingStatus');
    const countEl = document.getElementById('pointsCount');
    const zoneConfigForm = document.getElementById('zoneConfigForm');
    
    if (startBtn) startBtn.disabled = false;
    if (clearBtn) clearBtn.disabled = true;
    if (undoBtn) undoBtn.disabled = true;
    if (statusEl) {
        statusEl.textContent = 'Click "Start Drawing" to begin';
        statusEl.classList.remove('active');
    }
    if (countEl) countEl.textContent = 'Points: 0';
    
    // CRITICAL FIX: Keep form visible and remove hidden class if any
    if (zoneConfigForm) {
        zoneConfigForm.classList.remove('hidden');
        zoneConfigForm.style.display = 'block';
        zoneConfigForm.style.visibility = 'visible';
        
        // Reset form fields but keep form visible
        const zoneForm = document.getElementById('zoneForm');
        if (zoneForm) {
            zoneForm.reset();
        }
    }
    
    zoneDrawingSystem.canvas.classList.remove('drawing');
    
    redrawCanvas();
    
    console.log('✅ Zone creation finished - form kept visible');
}


// Cancel zone creation
function cancelZoneCreation() {
    finishZoneCreation();
    showNotification('Zone creation cancelled', 'info');
}

// Render zones list
function renderZonesList() {
    const zonesGrid = document.getElementById('zonesGrid');
    
    if (!zonesGrid) return;
    
    if (zoneDrawingSystem.zones.length === 0) {
        zonesGrid.innerHTML = `
            <div class="empty-zones">
                <i class="fas fa-draw-polygon"></i>
                <p>No zones configured yet</p>
                <p>Start by drawing your first detection zone</p>
            </div>
        `;
        return;
    }
    
    const priorityColors = {
        1: '#FF0000',
        2: '#FF8000',
        3: '#FFFF00',
        4: '#00FF00',
        5: '#0000FF'
    };
    
    zonesGrid.innerHTML = zoneDrawingSystem.zones.map(zone => `
        <div class="zone-card" style="--zone-color: ${zone.color}">
            <div class="zone-card-header">
                <div class="zone-info">
                    <h4>
                        <i class="fas fa-draw-polygon" style="color: ${zone.color}"></i>
                        ${zone.name}
                    </h4>
                    <span class="priority-badge" style="background: ${priorityColors[zone.priority]}">
                        Priority ${zone.priority}
                    </span>
                </div>
                <div class="zone-actions">
                    <button class="btn btn-outline btn-icon" onclick="editZone('${zone.id}')" title="Edit Zone">
                        <i class="fas fa-edit"></i>
                    </button>
                    <button class="btn btn-outline btn-icon" onclick="deleteZone('${zone.id}')" title="Delete Zone">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
            <div class="zone-details">
                <div class="detail-item">
                    <div class="label">Threshold</div>
                    <div class="value">${zone.threshold}</div>
                </div>
                <div class="detail-item">
                    <div class="label">Min Area</div>
                    <div class="value">${zone.min_area}px</div>
                </div>
                <div class="detail-item">
                    <div class="label">Motion Frames</div>
                    <div class="value">${zone.motion_frames}</div>
                </div>
                <div class="detail-item">
                    <div class="label">Points</div>
                    <div class="value">${zone.coordinates.length}</div>
                </div>
            </div>
        </div>
    `).join('');
}

// Delete zone
async function deleteZone(zoneId) {
    if (!confirm('Are you sure you want to delete this zone?')) return;
    
    try {
        // Delete from backend
        const response = await fetch(`${API_BASE}/api/zones/${zoneId}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            throw new Error('Failed to delete zone from server');
        }
        
        // Remove from local zones
        zoneDrawingSystem.zones = zoneDrawingSystem.zones.filter(zone => zone.id !== zoneId);
        
        // Save to local storage
        saveZonesToStorage();
        
        // Update UI
        renderZonesList();
        redrawCanvas();
        
        showNotification('Zone deleted successfully', 'success');
        
    } catch (error) {
        console.error('Error deleting zone:', error);
        showNotification('Failed to delete zone. Please try again.', 'error');
    }
}

// Edit zone (placeholder for future implementation)
function editZone(zoneId) {
    showNotification('Zone editing feature coming soon!', 'info');
}

// Export zones configuration
function exportZonesConfig() {
    const config = {
        zones: zoneDrawingSystem.zones,
        exported_at: new Date().toISOString(),
        version: '1.0'
    };
    
    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `zones_config_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    
    URL.revokeObjectURL(url);
    showNotification('Zones configuration exported successfully', 'success');
}

// Import zones configuration
function importZonesConfig(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            const config = JSON.parse(e.target.result);
            
            if (!config.zones || !Array.isArray(config.zones)) {
                throw new Error('Invalid configuration file format');
            }
            
            // Validate zones
            config.zones.forEach(zone => {
                if (!zone.name || !zone.coordinates || !Array.isArray(zone.coordinates)) {
                    throw new Error('Invalid zone data in configuration file');
                }
            });
            
            // Import zones
            zoneDrawingSystem.zones = config.zones;
            saveZonesToStorage();
            renderZonesList();
            redrawCanvas();
            
            showNotification(`Imported ${config.zones.length} zones successfully`, 'success');
            
        } catch (error) {
            console.error('Error importing zones:', error);
            showNotification('Failed to import zones configuration', 'error');
        }
    };
    
    reader.readAsText(file);
    event.target.value = ''; // Reset file input
}

// Storage functions
function saveZonesToStorage() {
    try {
        localStorage.setItem('radiguard_zones', JSON.stringify(zoneDrawingSystem.zones));
    } catch (error) {
        console.error('Error saving zones to storage:', error);
    }
}

function loadZonesFromStorage() {
    try {
        const stored = localStorage.getItem('radiguard_zones');
        if (stored) {
            zoneDrawingSystem.zones = JSON.parse(stored);
            renderZonesList();
        }
    } catch (error) {
        console.error('Error loading zones from storage:', error);
        zoneDrawingSystem.zones = [];
    }
}

// Show notifications
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <span>${message}</span>
        <button onclick="this.parentElement.remove()" style="background: none; border: none; color: inherit; font-size: 1.2rem; cursor: pointer; margin-left: 1rem;">&times;</button>
    `;

    // Add notification styles
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        z-index: 1000;
        animation: slideIn 0.3s ease;
        background: ${type === 'success' ? '#48bb78' : type === 'error' ? '#e53e3e' : type === 'warning' ? '#ed8936' : '#4299e1'};
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        display: flex;
        align-items: center;
        justify-content: space-between;
        min-width: 300px;
        max-width: 500px;
    `;

    // Add slide-in animation
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
    `;
    document.head.appendChild(style);

    document.body.appendChild(notification);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.style.animation = 'slideOut 0.3s ease forwards';
            setTimeout(() => {
                if (notification.parentElement) {
                    notification.remove();
                }
            }, 300);
        }
    }, 5000);

    // Add slide-out animation
    const slideOutStyle = document.createElement('style');
    slideOutStyle.textContent = `
        @keyframes slideOut {
            from {
                transform: translateX(0);
                opacity: 1;
            }
            to {
                transform: translateX(100%);
                opacity: 0;
            }
        }
    `;
    document.head.appendChild(slideOutStyle);
}

// Handle window resize for responsive behavior
window.addEventListener('resize', function() {
    const navLinks = document.getElementById('navLinks');
    const mobileMenuBtn = document.getElementById('mobileMenuBtn');
    
    // Close mobile menu on resize to desktop
    if (window.innerWidth > 768 && navLinks) {
        navLinks.classList.remove('active');
        if (mobileMenuBtn) {
            mobileMenuBtn.classList.remove('active');
            // Reset hamburger menu
            const spans = mobileMenuBtn.querySelectorAll('span');
            spans.forEach(span => {
                span.style.transform = 'none';
                span.style.opacity = '1';
            });
        }
    }
});

// Utility function for debouncing
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Smooth scroll to top function
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Additional error handling for API calls
window.addEventListener('unhandledrejection', function(event) {
    console.error('Unhandled promise rejection:', event.reason);
    showNotification('An unexpected error occurred. Please try again.', 'error');
});

// Performance monitoring
if ('performance' in window) {
    window.addEventListener('load', function() {
        setTimeout(function() {
            const perfData = performance.getEntriesByType('navigation')[0];
            console.log('Page load time:', perfData.loadEventEnd - perfData.loadEventStart, 'ms');
        }, 0);
    });
}

// Monitor and maintain form visibility
function ensureFormVisibility() {
    const zoneConfigForm = document.getElementById('zoneConfigForm');
    if (zoneConfigForm) {
        // Remove any hidden classes
        zoneConfigForm.classList.remove('hidden');
        
        // Force visibility styles
        zoneConfigForm.style.display = 'block';
        zoneConfigForm.style.visibility = 'visible';
        zoneConfigForm.style.opacity = '1';
    }
}

// Call this function after any zone operation
function callEnsureFormVisibility() {
    setTimeout(ensureFormVisibility, 100);
}
