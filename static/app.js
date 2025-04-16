/**
 * VeinVision Pro - Main Application JavaScript
 * Enhanced version with improved UI/UX and new features
 */

// Global variables
const settings = {
    detection_method: 'adaptive',
    led_brightness: 128,
    led_pattern: 1,
    camera_exposure: 20000,
    camera_gain: 10,
    zoom_level: 1.0,
    rotation: 0,
    clahe_clip_limit: 5.0,
    clahe_tile_grid_size: 8,
    frangi_scale_min: 1.0,
    frangi_scale_max: 5.0,
    frangi_scale_step: 1.0,
    frangi_beta: 0.5,
    frangi_gamma: 15
};

// Gallery pagination state
const galleryState = {
    currentPage: 1,
    pageSize: 12,
    totalPages: 1,
    images: [],
    filteredImages: [],
    searchQuery: ''
};

// Last captured image state
let lastCapturedImage = {
    processed: null,
    original: null,
    timestamp: null
};

$(document).ready(function() {
    // Initialize Bootstrap modals
    const modals = document.querySelectorAll('.modal');
    modals.forEach(modalEl => {
        new bootstrap.Modal(modalEl);
    });

    // Load settings initially
    fetch('/get_settings')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                Object.assign(settings, data.settings);
                
                // Update UI controls
                updateUIFromSettings();
                updateCameraInfo();
            }
        })
        .catch(error => {
            console.error('Error loading settings:', error);
            showAlert('Error loading settings. Check console for details.', 'danger');
        });

    // Initialize all range sliders
    initializeRangeSliders();
    
    // Apply camera settings button
    $('#applyCameraSettings').on('click', function(e) {
        e.preventDefault();
        applySettings();
    });
    
    // Stream resolution selector change
    $('#streamResolution').on('change', function() {
        changeResolution(this.value);
    });
    
    // Capture image button
    $('#captureBtn').on('click', captureImage);
    
    // Toggle Last Captured Image close button
    $('#closeLastCapturedBtn').on('click', function() {
        $('#lastCapturedContainer').addClass('d-none');
    });
    
    // Toggle Comparison Container close button
    $('#closeComparisonBtn').on('click', function() {
        $('#comparisonContainer').addClass('d-none');
    });
    
    // Image toggle buttons for last captured
    $('#toggleOriginalBtn').on('click', function() {
        if (!lastCapturedImage.original) return;
        
        $('#lastCapturedImg').attr('src', lastCapturedImage.original + '?t=' + new Date().getTime());
        $(this).addClass('active');
        $('#toggleProcessedBtn').removeClass('active');
    });
    
    $('#toggleProcessedBtn').on('click', function() {
        if (!lastCapturedImage.processed) return;
        
        $('#lastCapturedImg').attr('src', lastCapturedImage.processed + '?t=' + new Date().getTime());
        $(this).addClass('active');
        $('#toggleOriginalBtn').removeClass('active');
    });

    // Zoom buttons
    $('#zoomInBtn').on('click', function() {
        zoomImage('in');
    });
    
    $('#zoomOutBtn').on('click', function() {
        zoomImage('out');
    });
    
    // Rotate button
    $('#rotateBtn').on('click', rotateImage);
    
    // Compare button
    $('#compareBtn').on('click', toggleComparisonView);
    $('#refreshComparisonBtn').on('click', refreshComparisonView);
    
    // LED controls
    $('#toggleLEDs').on('click', toggleLEDs);
    $('#applyLEDSettings').on('click', applyLEDSettings);
    
    // Patient form submission
    $('#patientForm').on('submit', function(e) {
        e.preventDefault();
        savePatientInfo();
    });
    
    // Notes functionality
    $('#saveNotesBtn').on('click', saveNotes);
    $('#clearNotesBtn').on('click', function() {
        $('#procedureNotes').val('');
        showAlert('Notes cleared', 'info');
    });
    
    // Fullscreen button
    $('#fullscreenBtn').on('click', toggleFullscreen);
    
    // Gallery controls
    $('#refreshGalleryBtn').on('click', refreshGallery);
    $('#clearGalleryBtn').on('click', clearGallery);
    $('#gallerySearch').on('input', searchGallery);
    $('#gallerySearchBtn').on('click', searchGallery);
    $('#galleryPageSize').on('change', changePageSize);
    $('#paginationPrev').on('click', prevGalleryPage);
    $('#paginationNext').on('click', nextGalleryPage);
    
    // Dark mode toggle
    $('#darkModeToggle').on('click', toggleDarkMode);
    
    // Shutdown confirmation
    $('#confirmShutdown').on('click', confirmShutdown);
    
    // Modal image controls
    $('#modalViewOriginal').on('click', function() {
        const src = $(this).attr('data-src');
        $('#modalImage').attr('src', src + '?t=' + new Date().getTime());
        
        // Update active state
        $(this).addClass('active');
        $('#modalViewProcessed').removeClass('active');
        
        // Update download link
        $('#modalDownload').attr('data-src', src);
        $('#modalDownload').attr('href', src);
    });
    
    $('#modalViewProcessed').on('click', function() {
        const src = $(this).attr('data-src');
        $('#modalImage').attr('src', src + '?t=' + new Date().getTime());
        
        // Update active state
        $(this).addClass('active');
        $('#modalViewOriginal').removeClass('active');
        
        // Update download link
        $('#modalDownload').attr('data-src', src);
        $('#modalDownload').attr('href', src);
    });
    
    // Handle download button in modal
    $('#modalDownload').on('click', function(e) {
        e.preventDefault();
        const imgSrc = $(this).attr('data-src');
        
        // Create a temporary link and trigger download
        const link = document.createElement('a');
        link.href = imgSrc;
        link.download = imgSrc.split('/').pop();
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });
    
    // Print button
    $('#modalPrintBtn').on('click', function() {
        const imgSrc = $('#modalImage').attr('src');
        const timestamp = $('#modalTimestamp').text();
        
        // Create a print window with the image and metadata
        const printWin = window.open('', '_blank', 'width=800,height=600');
        printWin.document.write(`
            <html>
            <head>
                <title>VeinVision Pro - Image Print</title>
                <style>
                    body { font-family: Arial, sans-serif; text-align: center; }
                    .image-container { margin: 20px auto; max-width: 90%; }
                    img { max-width: 100%; height: auto; }
                    .metadata { margin-top: 20px; text-align: left; display: inline-block; border-top: 1px solid #ccc; padding-top: 10px; }
                    table { border-collapse: collapse; width: 100%; }
                    th { text-align: right; padding: 5px 10px; }
                    td { text-align: left; padding: 5px 10px; }
                </style>
            </head>
            <body>
                <h1>VeinVision Pro - Vein Image</h1>
                <div class="image-container">
                    <img src="${imgSrc}" alt="Vein image">
                </div>
                <div class="metadata">
                    <table>
                        <tr>
                            <th>Timestamp:</th>
                            <td>${timestamp}</td>
                        </tr>
                        <tr>
                            <th>Detection Method:</th>
                            <td>${$('#modalDetectionMethod').text()}</td>
                        </tr>
                        <tr>
                            <th>Patient Name:</th>
                            <td>${$('#modalPatientName').text()}</td>
                        </tr>
                        <tr>
                            <th>Patient ID:</th>
                            <td>${$('#modalPatientId').text()}</td>
                        </tr>
                        <tr>
                            <th>Procedure:</th>
                            <td>${$('#modalProcedure').text()}</td>
                        </tr>
                    </table>
                </div>
                <footer>
                    <p>VeinVision Pro - Advanced Medical Imaging System</p>
                </footer>
                <script>
                    window.onload = function() { window.print(); }
                </script>
            </body>
            </html>
        `);
        printWin.document.close();
    });
    
    // Calibration wizard controls
    $('.next-cal-step').on('click', function() {
        const nextStepId = $(this).data('next');
        $('.calibration-step').addClass('d-none');
        $('#' + nextStepId).removeClass('d-none');
    });
    
    $('.prev-cal-step').on('click', function() {
        const prevStepId = $(this).data('prev');
        $('.calibration-step').addClass('d-none');
        $('#' + prevStepId).removeClass('d-none');
    });
    
    $('#finishCalibration').on('click', function() {
        const detectionMethod = $('input[name="calDetectionMethod"]:checked').val();
        const brightness = $('#calLedBrightness').val();
        const exposure = $('#calExposure').val();
        const gain = $('#calGain').val();
        
        // Apply all settings at once
        const updatedSettings = {
            detection_method: detectionMethod,
            led_brightness: parseInt(brightness),
            camera_exposure: parseInt(exposure),
            camera_gain: parseFloat(gain)
        };
        
        fetch('/update_settings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(updatedSettings)
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Update our local settings
                Object.assign(settings, updatedSettings);
                
                // Update UI
                updateUIFromSettings();
                
                // Close the modal
                const modal = bootstrap.Modal.getInstance(document.getElementById('calibrationModal'));
                modal.hide();
                
                showAlert('Calibration complete! Settings have been applied.', 'success');
            } else {
                showAlert('Error applying calibration settings', 'danger');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showAlert('Error communicating with server', 'danger');
        });
    });
    
    // Initialize hardware status
    updateHardwareStatus();
    
    // Update capture counter
    updateCaptureCounter();
    
    // Initialize gallery
    loadImageGallery();
    
    // Check for saved dark mode preference
    if (localStorage.getItem('darkMode') === 'enabled') {
        document.body.classList.add('dark-mode');
        $('#darkModeToggle').html('<i class="fas fa-sun me-2"></i>Light Mode');
    }
    
    // Check if this was a first load, show welcome message
    if (!localStorage.getItem('hasVisited')) {
        setTimeout(function() {
            showAlert('Welcome to VeinVision Pro!', 'info', 10000);
            localStorage.setItem('hasVisited', 'true');
        }, 1000);
    }
});

// FUNCTION DEFINITIONS

// Initialize all range sliders
function initializeRangeSliders() {
    // LED Brightness slider
    const ledBrightness = document.getElementById('ledBrightness');
    const brightnessValue = document.getElementById('brightnessValue');
    
    if (ledBrightness && brightnessValue) {
        ledBrightness.addEventListener('input', function() {
            brightnessValue.textContent = this.value;
        });
    }
    
    // Camera Exposure slider
    const cameraExposure = document.getElementById('cameraExposure');
    const exposureValue = document.getElementById('exposureValue');
    
    if (cameraExposure && exposureValue) {
        cameraExposure.addEventListener('input', function() {
            exposureValue.textContent = this.value;
        });
    }
    
    // Camera Gain slider
    const cameraGain = document.getElementById('cameraGain');
    const gainValue = document.getElementById('gainValue');
    
    if (cameraGain && gainValue) {
        cameraGain.addEventListener('input', function() {
            gainValue.textContent = this.value;
        });
    }
    
    // CLAHE Clip Limit slider
    const claheClipLimit = document.getElementById('claheClipLimit');
    const clipLimitValue = document.getElementById('clipLimitValue');
    
    if (claheClipLimit && clipLimitValue) {
        claheClipLimit.addEventListener('input', function() {
            clipLimitValue.textContent = this.value;
        });
    }
    
    // CLAHE Tile Size slider
    const claheTileSize = document.getElementById('claheTileSize');
    const tileSizeValue = document.getElementById('tileSizeValue');
    
    if (claheTileSize && tileSizeValue) {
        claheTileSize.addEventListener('input', function() {
            tileSizeValue.textContent = this.value;
        });
    }
}

// Update UI controls from settings object
function updateUIFromSettings() {
    // Detection method
    $('#detectionMethod').val(settings.detection_method);
    $('#detectionMethodDisplay').text(capitalizeFirstLetter(settings.detection_method));
    
    // Camera settings
    $('#cameraExposure').val(settings.camera_exposure);
    $('#exposureValue').text(settings.camera_exposure);
    $('#exposureInfo').text(`Exp: ${settings.camera_exposure}μs`);
    
    $('#cameraGain').val(settings.camera_gain);
    $('#gainValue').text(settings.camera_gain);
    $('#gainInfo').text(`Gain: ${settings.camera_gain}`);
    
    // Update zoom display
    $('#zoomInfo').text(`Zoom: ${settings.zoom_level.toFixed(1)}x`);
    
    // LED settings
    $('#ledBrightness').val(settings.led_brightness);
    $('#brightnessValue').text(settings.led_brightness);
    $('#ledPattern').val(settings.led_pattern);
    
    // CLAHE settings
    if ($('#claheClipLimit').length) {
        $('#claheClipLimit').val(settings.clahe_clip_limit);
        $('#clipLimitValue').text(settings.clahe_clip_limit);
    }
    
    if ($('#claheTileSize').length) {
        $('#claheTileSize').val(settings.clahe_tile_grid_size);
        $('#tileSizeValue').text(settings.clahe_tile_grid_size);
    }
    
    // Stream resolution
    $('#streamResolution').val(settings.stream_resolution);
}

// Apply camera and detection settings
function applySettings() {
    const updatedSettings = {
        camera_exposure: parseInt($('#cameraExposure').val()),
        camera_gain: parseFloat($('#cameraGain').val()),
        detection_method: $('#detectionMethod').val(),
        stream_resolution: $('#streamResolution').val()
    };
    
    // Add CLAHE settings if they exist
    if ($('#claheClipLimit').length) {
        updatedSettings.clahe_clip_limit = parseFloat($('#claheClipLimit').val());
    }
    
    if ($('#claheTileSize').length) {
        updatedSettings.clahe_tile_grid_size = parseInt($('#claheTileSize').val());
    }

    Object.assign(settings, updatedSettings);

    fetch('/update_settings', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(updatedSettings)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showAlert('Camera settings updated successfully', 'success');
            updateCameraInfo();
            
            // Update method display in the system stats
            $('#detectionMethodDisplay').text(capitalizeFirstLetter(settings.detection_method));
        } else {
            showAlert('Error updating camera settings', 'danger');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('Error communicating with server', 'danger');
    });
}

// Change stream resolution
function changeResolution(newResolution) {
    // Show loading indicator
    const videoStream = document.querySelector('.video-stream');
    videoStream.style.opacity = '0.5';

    fetch('/update_settings', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            stream_resolution: newResolution
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showAlert(`Stream resolution changed to ${newResolution}`, 'success');

            // Reload the video stream
            const videoSrc = videoStream.src;
            videoStream.src = '';
            setTimeout(() => {
                videoStream.src = videoSrc + '?t=' + new Date().getTime();
                videoStream.style.opacity = '1';
            }, 1000);

            // Update settings
            settings.stream_resolution = newResolution;
        } else {
            showAlert('Error changing resolution', 'danger');
            videoStream.style.opacity = '1';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('Error communicating with server', 'danger');
        videoStream.style.opacity = '1';
    });
}

// Capture image function
function captureImage() {
    // Show loading indicator
    const captureBtn = document.getElementById('captureBtn');
    const captureSpinner = document.getElementById('captureSpinner');
    const captureBtnText = document.getElementById('captureBtnText');

    captureBtn.disabled = true;
    captureSpinner.classList.remove('d-none');
    captureBtnText.textContent = 'Processing...';

    fetch('/capture', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            patient_info: {
                name: $('#patientName').val() || 'unknown',
                id: $('#patientId').val() || '',
                age: $('#patientAge').val() || '',
                procedure: $('#procedureType').val() || ''
            },
            detection_method: settings.detection_method
        })
    })
    .then(response => response.json())
    .then(data => {
        // Hide loading indicator
        captureBtn.disabled = false;
        captureSpinner.classList.add('d-none');
        captureBtnText.textContent = 'Capture';

        if (data.success) {
            showAlert('Image captured successfully', 'success');

            // Save the captured images for later use
            lastCapturedImage = {
                processed: data.processed_image,
                original: data.original_image,
                timestamp: data.timestamp
            };
            
            // Update the last captured image
            const lastCapturedImg = document.getElementById('lastCapturedImg');
            lastCapturedImg.src = data.processed_image + '?t=' + new Date().getTime();

            // Show the last captured image container
            document.getElementById('lastCapturedContainer').classList.remove('d-none');

            // Update captured image info alert
            const capturedImageInfo = document.getElementById('capturedImageInfo');
            capturedImageInfo.classList.remove('d-none');

            // Format and set timestamp
            const captureTimestamp = document.getElementById('captureTimestamp');
            try {
                const timestamp = data.timestamp;
                const date = new Date(
                    timestamp.substring(0, 4),
                    parseInt(timestamp.substring(4, 6)) - 1,
                    timestamp.substring(6, 8),
                    timestamp.substring(9, 11),
                    timestamp.substring(11, 13),
                    timestamp.substring(13, 15)
                );
                captureTimestamp.textContent = date.toLocaleString();
            } catch (e) {
                captureTimestamp.textContent = data.timestamp || 'Unknown';
            }

            // Set view buttons
            const viewOriginalBtn = document.getElementById('viewOriginalBtn');
            const viewProcessedBtn = document.getElementById('viewProcessedBtn');
            const downloadBtn = document.getElementById('downloadBtn');

            viewOriginalBtn.href = data.original_image;
            viewProcessedBtn.href = data.processed_image;
            downloadBtn.href = data.processed_image;
            downloadBtn.download = data.processed_image.split('/').pop();

            // Reset toggle buttons
            $('#toggleProcessedBtn').addClass('active');
            $('#toggleOriginalBtn').removeClass('active');
            
            // Update counter
            updateCaptureCounter();

            // Refresh gallery
            loadImageGallery();
        } else {
            showAlert('Error capturing image: ' + data.message, 'danger');
        }
    })
    .catch(error => {
        // Hide loading indicator
        captureBtn.disabled = false;
        captureSpinner.classList.add('d-none');
        captureBtnText.textContent = 'Capture';

        console.error('Error:', error);
        showAlert('Error communicating with server', 'danger');
    });
}

// Show method comparison view
function toggleComparisonView() {
    const comparisonContainer = $('#comparisonContainer');
    
    if (comparisonContainer.hasClass('d-none')) {
        comparisonContainer.removeClass('d-none');
        refreshComparisonView();
    } else {
        comparisonContainer.addClass('d-none');
    }
}

// Refresh the comparison view with the latest images
function refreshComparisonView() {
    // Set placeholders for comparison images
    $('#compareAdaptive').attr('src', '{{ url_for("static", filename="images/placeholder.jpg") }}');
    $('#compareFrangi').attr('src', '{{ url_for("static", filename="images/placeholder.jpg") }}');
    $('#compareLaplacian').attr('src', '{{ url_for("static", filename="images/placeholder.jpg") }}');
    
    // Make separate requests for each method
    const methods = ['adaptive', 'frangi', 'laplacian'];
    const requests = methods.map(method => {
        return fetch('/capture', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                detection_method: method,
                for_comparison: true
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Update the corresponding image
                $(`#compare${capitalizeFirstLetter(method)}`).attr('src', data.processed_image + '?t=' + new Date().getTime());
                return true;
            } else {
                console.error(`Error capturing ${method} image:`, data.message);
                return false;
            }
        })
        .catch(error => {
            console.error(`Error in ${method} request:`, error);
            return false;
        });
    });
    
    // After all requests are done
    Promise.all(requests)
        .then(results => {
            if (results.some(success => success)) {
                showAlert('Method comparison refreshed', 'success');
            } else {
                showAlert('Error generating method comparison', 'danger');
            }
        });
}

// Zoom in/out function
function zoomImage(action) {
    fetch('/zoom', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ action: action })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Update display
            settings.zoom_level = data.zoom_level;
            $('#zoomInfo').text(`Zoom: ${settings.zoom_level.toFixed(1)}x`);
            showAlert(`Zoomed ${action}`, 'info');
        } else {
            showAlert('Zoom error', 'danger');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('Error communicating with server', 'danger');
    });
}

// Rotate image function
function rotateImage() {
    fetch('/rotate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            settings.rotation = data.rotation;
            showAlert(`Image rotated to ${settings.rotation}°`, 'info');
        } else {
            showAlert('Rotation error', 'danger');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('Error communicating with server', 'danger');
    });
}

// Toggle LEDs function
function toggleLEDs() {
    fetch('/toggle_leds', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const ledStatus = document.getElementById('ledStatus');
            ledStatus.textContent = data.led_status ? 'Active' : 'Inactive';
            showAlert(`LEDs ${data.led_status ? 'activated' : 'deactivated'} successfully`, 'success');
        } else {
            showAlert('Error toggling LEDs', 'danger');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('Error communicating with server', 'danger');
    });
}

// Apply LED settings function
function applyLEDSettings() {
    const updatedSettings = {
        led_brightness: parseInt(document.getElementById('ledBrightness').value),
        led_pattern: parseInt(document.getElementById('ledPattern').value)
    };

    Object.assign(settings, updatedSettings);

    fetch('/update_settings', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(updatedSettings)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showAlert('LED settings updated successfully', 'success');
        } else {
            showAlert('Error updating LED settings', 'danger');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('Error communicating with server', 'danger');
    });
}

// Save patient information
function savePatientInfo() {
    const patientData = {
        name: $('#patientName').val(),
        id: $('#patientId').val(),
        age: $('#patientAge').val(),
        procedure: $('#procedureType').val()
    };

    fetch('/save_patient', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(patientData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showAlert('Patient information saved successfully', 'success');
        } else {
            showAlert('Error saving patient information', 'danger');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('Error communicating with server', 'danger');
    });
}

// Save notes function
function saveNotes() {
    const notes = $('#procedureNotes').val();

    fetch('/save_notes', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ notes: notes })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showAlert('Notes saved successfully', 'success');
        } else {
            showAlert('Error saving notes', 'danger');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('Error communicating with server', 'danger');
    });
}

// Toggle fullscreen function
function toggleFullscreen() {
    const videoContainer = document.querySelector('.video-container');

    if (!document.fullscreenElement) {
        if (videoContainer.requestFullscreen) {
            videoContainer.requestFullscreen();
        } else if (videoContainer.webkitRequestFullscreen) {
            videoContainer.webkitRequestFullscreen();
        } else if (videoContainer.msRequestFullscreen) {
            videoContainer.msRequestFullscreen();
        }
        $('#fullscreenBtn').html('<i class="fas fa-compress"></i>');
    } else {
        if (document.exitFullscreen) {
            document.exitFullscreen();
        } else if (document.webkitExitFullscreen) {
            document.webkitExitFullscreen();
        } else if (document.msExitFullscreen) {
            document.msExitFullscreen();
        }
        $('#fullscreenBtn').html('<i class="fas fa-expand"></i>');
    }
}

// Gallery functions
function loadImageGallery() {
    const galleryContainer = $('#imageGallery');
    
    // Show loading spinner
    galleryContainer.html(`
        <div class="col-12 text-center py-5">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-2">Loading saved images...</p>
        </div>
    `);
    
    // Add cache-busting parameter to avoid cached responses
    const cacheBuster = new Date().getTime();
    
    // Get page size from select
    const pageSize = $('#galleryPageSize').val() || galleryState.pageSize;
    
    fetch(`/images?_=${cacheBuster}&page=${galleryState.currentPage}&size=${pageSize}&query=${galleryState.searchQuery}`)
        .then(response => response.json())
        .then(data => {
            if (data.success && data.images && data.images.length > 0) {
                console.log(`Found ${data.images.length} images`);
                
                // Store images in state
                galleryState.images = data.images;
                galleryState.totalPages = Math.ceil(data.images.length / pageSize);
                updateGalleryDisplay();
            } else {
                galleryContainer.html(`
                    <div class="col-12 text-center py-5">
                        <i class="fas fa-images fa-3x mb-3 text-muted"></i>
                        <p>No images captured yet. Use the capture button to save images.</p>
                    </div>
                `);
                console.log('No images found or empty response');
                
                // Reset state
                galleryState.images = [];
                galleryState.totalPages = 1;
                galleryState.currentPage = 1;
                updatePaginationDisplay();
            }
        })
        .catch(error => {
            console.error('Error loading gallery:', error);
            galleryContainer.html(`
                <div class="col-12 text-center py-5">
                    <i class="fas fa-exclamation-triangle fa-3x mb-3 text-danger"></i>
                    <p>Error loading gallery. Please try refreshing.</p>
                    <p class="small text-muted">Error: ${error.message}</p>
                </div>
            `);
        });
}

// Update gallery display based on current state
function updateGalleryDisplay() {
    const galleryContainer = $('#imageGallery');
    const pageSize = parseInt($('#galleryPageSize').val() || galleryState.pageSize);
    
    // Calculate which images to show
    const startIdx = (galleryState.currentPage - 1) * pageSize;
    const endIdx = Math.min(startIdx + pageSize, galleryState.images.length);
    const imagesToDisplay = galleryState.images.slice(startIdx, endIdx);
    
    // Update pagination info
    galleryState.totalPages = Math.ceil(galleryState.images.length / pageSize);
    updatePaginationDisplay();
    
    if (imagesToDisplay.length === 0) {
        galleryContainer.html(`
            <div class="col-12 text-center py-5">
                <i class="fas fa-search fa-3x mb-3 text-muted"></i>
                <p>No matching images found.</p>
            </div>
        `);
        return;
    }
    
    let galleryHTML = '';
    const cacheBuster = new Date().getTime();
    
    imagesToDisplay.forEach(image => {
        const timestamp = image.timestamp;
        let formattedDate = '';
        try {
            // Parse timestamp format: YYYYMMDD_HHMMSS
            const year = timestamp.substring(0, 4);
            const month = parseInt(timestamp.substring(4, 6)) - 1; // JS months are 0-based
            const day = timestamp.substring(6, 8);
            const hour = timestamp.substring(9, 11);
            const minute = timestamp.substring(11, 13);
            const second = timestamp.substring(13, 15);

            const date = new Date(year, month, day, hour, minute, second);
            formattedDate = date.toLocaleString();
        } catch (e) {
            console.error("Error parsing date:", e);
            formattedDate = timestamp || 'Unknown';
        }

        // Add cache-busting parameter to image URLs
        const processedImage = `${image.processed_image}?t=${cacheBuster}`;
        const originalImage = image.original_image ? `${image.original_image}?t=${cacheBuster}` : null;

        galleryHTML += `
            <div class="col-md-4 col-sm-6 mb-4">
                <div class="card image-card h-100">
                    <div class="position-relative">
                        <img src="${processedImage}" 
                             class="card-img-top gallery-image" 
                             alt="Captured Image"
                             data-img-src="${image.processed_image}"
                             data-orig-src="${image.original_image || ''}"
                             data-timestamp="${timestamp}">
                        <div class="card-overlay">
                            <small>${formattedDate}</small>
                        </div>
                    </div>
                    <div class="card-footer d-flex justify-content-between">
                        <button class="btn btn-sm btn-primary view-image-btn" 
                                data-processed-src="${image.processed_image}" 
                                data-original-src="${image.original_image || ''}" 
                                data-timestamp="${timestamp}">
                            <i class="fas fa-eye me-1"></i>View
                        </button>
                        <button class="btn btn-sm btn-danger delete-image-btn" 
                                data-image-path="${image.processed_image}">
                            <i class="fas fa-trash me-1"></i>Delete
                        </button>
                    </div>
                </div>
            </div>
        `;
    });
    
    galleryContainer.html(galleryHTML);
    
    // Add event handlers
    addGalleryEventHandlers();
}

// Add event handlers to gallery items
function addGalleryEventHandlers() {
    // Image view buttons
    $('.view-image-btn').on('click', function() {
        const processedSrc = $(this).data('processed-src');
        const originalSrc = $(this).data('original-src');
        const timestamp = $(this).data('timestamp');

        // Set image source in modal with cache-buster
        $('#modalImage').attr('src', processedSrc + '?t=' + new Date().getTime());
        
        // Format timestamp for display
        let formattedDate = '';
        try {
            // Parse timestamp format: YYYYMMDD_HHMMSS
            const year = timestamp.substring(0, 4);
            const month = parseInt(timestamp.substring(4, 6)) - 1; // JS months are 0-based
            const day = timestamp.substring(6, 8);
            const hour = timestamp.substring(9, 11);
            const minute = timestamp.substring(11, 13);
            const second = timestamp.substring(13, 15);

            const date = new Date(year, month, day, hour, minute, second);
            formattedDate = date.toLocaleString();
        } catch (e) {
            formattedDate = timestamp || 'Unknown';
        }
        
        // Update metadata
        $('#modalTimestamp').text(formattedDate);
        $('#modalDetectionMethod').text(capitalizeFirstLetter(settings.detection_method));
        $('#modalPatientName').text($('#patientName').val() || '-');
        $('#modalPatientId').text($('#patientId').val() || '-');
        $('#modalProcedure').text($('#patientAge').val() ? `${$('#procedureType').val()} (Age: ${$('#patientAge').val()})` : $('#procedureType').val() || '-');
        $('#modalResolution').text(settings.stream_resolution || '-');

        // Update toggle buttons
        const viewOriginalBtn = $('#modalViewOriginal');
        const viewProcessedBtn = $('#modalViewProcessed');

        // Enable/disable original button based on source availability
        if (originalSrc) {
            viewOriginalBtn.attr('data-src', originalSrc);
            viewOriginalBtn.prop('disabled', false);
            
            // Set click event handlers directly here to ensure they're properly attached
            viewOriginalBtn.off('click').on('click', function() {
                const src = $(this).attr('data-src');
                if (src) {
                    $('#modalImage').attr('src', src + '?t=' + new Date().getTime());
                    
                    // Update active state
                    $(this).addClass('active');
                    $('#modalViewProcessed').removeClass('active');
                    
                    // Update download link
                    $('#modalDownload').attr('data-src', src);
                    $('#modalDownload').attr('href', src);
                }
            });
        } else {
            viewOriginalBtn.attr('data-src', '');
            viewOriginalBtn.prop('disabled', true);
        }
        viewProcessedBtn.attr('data-src', processedSrc);
        
        // Set processed button click event
        viewProcessedBtn.off('click').on('click', function() {
            const src = $(this).attr('data-src');
            $('#modalImage').attr('src', src + '?t=' + new Date().getTime());
            
            // Update active state
            $(this).addClass('active');
            $('#modalViewOriginal').removeClass('active');
            
            // Update download link
            $('#modalDownload').attr('data-src', src);
            $('#modalDownload').attr('href', src);
        });

        // Set active state
        viewProcessedBtn.addClass('active');
        viewOriginalBtn.removeClass('active');

        // Set download link
        $('#modalDownload').attr('data-src', processedSrc);
        $('#modalDownload').attr('href', processedSrc);
        $('#modalDownload').attr('download', processedSrc.split('/').pop().split('?')[0]);

        // Show the modal
        const imageModal = new bootstrap.Modal(document.getElementById('imageModal'));
        imageModal.show();
    });
    
    // Image delete buttons
    $('.delete-image-btn').on('click', function() {
        const imagePath = $(this).data('image-path');
        const imageCard = $(this).closest('.col-md-4');

        if (confirm('Are you sure you want to delete this image?')) {
            console.log('Sending delete request for:', imagePath);

            fetch('/delete_image', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image_path: imagePath })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Remove the image from the gallery
                    imageCard.remove();
                    
                    // Update internal state
                    galleryState.images = galleryState.images.filter(img => 
                        img.processed_image !== imagePath
                    );
                    
                    // Update pagination if needed
                    if (galleryState.images.length === 0 && galleryState.currentPage > 1) {
                        galleryState.currentPage--;
                    }
                    
                    // Refresh gallery
                    updateGalleryDisplay();
                    updateCaptureCounter();
                    
                    showAlert('Image deleted successfully', 'success');
                    console.log('Delete response:', data);
                } else {
                    showAlert('Error deleting image: ' + data.message, 'danger');
                    console.error('Delete error:', data.message);
                }
            })
            .catch(error => {
                console.error('Error during delete request:', error);
                showAlert('Error communicating with server', 'danger');
            });
        }
    });
    
    // Make gallery images clickable to open modal
    $('.gallery-image').on('click', function() {
        const processedSrc = $(this).data('img-src');
        const originalSrc = $(this).data('orig-src');
        const timestamp = $(this).data('timestamp');
        
        // Trigger the view button click
        $(this).closest('.card').find('.view-image-btn').trigger('click');
    });
}

// Update pagination display
function updatePaginationDisplay() {
    $('#paginationInfo').text(`Page ${galleryState.currentPage} of ${galleryState.totalPages}`);
    
    // Enable/disable prev/next buttons
    $('#paginationPrev').prop('disabled', galleryState.currentPage <= 1);
    $('#paginationNext').prop('disabled', galleryState.currentPage >= galleryState.totalPages);
}

// Previous page
function prevGalleryPage() {
    if (galleryState.currentPage > 1) {
        galleryState.currentPage--;
        updateGalleryDisplay();
    }
}

// Next page
function nextGalleryPage() {
    if (galleryState.currentPage < galleryState.totalPages) {
        galleryState.currentPage++;
        updateGalleryDisplay();
    }
}

// Change page size
function changePageSize() {
    const newPageSize = parseInt($('#galleryPageSize').val());
    galleryState.pageSize = newPageSize;
    galleryState.currentPage = 1; // Reset to first page
    updateGalleryDisplay();
}

// Search gallery
function searchGallery() {
    galleryState.searchQuery = $('#gallerySearch').val();
    galleryState.currentPage = 1; // Reset to first page
    
    // If we're searching within already loaded images
    if (galleryState.images.length > 0 && galleryState.searchQuery) {
        // Filter images by timestamp or any other metadata
        galleryState.filteredImages = galleryState.images.filter(image => {
            // Convert timestamp to date string for searching
            let dateStr = '';
            try {
                const timestamp = image.timestamp;
                const year = timestamp.substring(0, 4);
                const month = parseInt(timestamp.substring(4, 6)) - 1;
                const day = timestamp.substring(6, 8);
                const date = new Date(year, month, day);
                dateStr = date.toLocaleDateString();
            } catch (e) {
                dateStr = image.timestamp || '';
            }
            
            // Search in date string
            return dateStr.toLowerCase().includes(galleryState.searchQuery.toLowerCase());
        });
        
        // Update display with filtered images
        updateGalleryDisplay();
    } else {
        // Reload from server with search query
        loadImageGallery();
    }
}

// Refresh gallery
function refreshGallery() {
    // Reset search and pagination
    galleryState.searchQuery = '';
    galleryState.currentPage = 1;
    $('#gallerySearch').val('');
    
    // Reload images
    loadImageGallery();
    showAlert('Gallery refreshed', 'info');
}

// Clear gallery
function clearGallery() {
    if (confirm('Are you sure you want to clear all saved images? This cannot be undone.')) {
        fetch('/clear_gallery', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showAlert('Gallery cleared successfully', 'success');
                
                // Reset state and reload
                galleryState.images = [];
                galleryState.filteredImages = [];
                galleryState.currentPage = 1;
                galleryState.totalPages = 1;
                
                loadImageGallery();
                updateCaptureCounter();
            } else {
                showAlert('Error clearing gallery', 'danger');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showAlert('Error communicating with server', 'danger');
        });
    }
}

// Toggle dark mode
function toggleDarkMode() {
    document.body.classList.toggle('dark-mode');

    if (document.body.classList.contains('dark-mode')) {
        $('#darkModeToggle').html('<i class="fas fa-sun me-2"></i>Light Mode');
        localStorage.setItem('darkMode', 'enabled');
    } else {
        $('#darkModeToggle').html('<i class="fas fa-moon me-2"></i>Dark Mode');
        localStorage.setItem('darkMode', 'disabled');
    }
}

// System shutdown
function confirmShutdown() {
    fetch('/shutdown', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showAlert('System is shutting down...', 'warning');
            // Close the modal
            const shutdownModal = bootstrap.Modal.getInstance(document.getElementById('shutdownModal'));
            shutdownModal.hide();
            // Show a full page message
            document.body.innerHTML = `
                <div class="d-flex justify-content-center align-items-center vh-100">
                    <div class="text-center">
                        <i class="fas fa-power-off fa-5x text-danger mb-4"></i>
                        <h1>System Shutting Down</h1>
                        <p class="lead">The Raspberry Pi is now shutting down safely.</p>
                        <p>Please wait until the system is fully powered off before disconnecting power.</p>
                        <div class="spinner-border text-primary mt-4" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                    </div>
                </div>
            `;
        } else {
            showAlert('Error shutting down system: ' + data.message, 'danger');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('Error communicating with server', 'danger');
    });
}

// Helper Functions

// Update camera info display
function updateCameraInfo() {
    fetch('/camera_info')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Update exposure and gain info in the camera-info div
                $('#exposureInfo').text(`Exp: ${data.camera_info.exposure}μs`);
                $('#gainInfo').text(`Gain: ${data.camera_info.gain}`);

                // Update detection method display
                $('#detectionMethodDisplay').text(capitalizeFirstLetter(settings.detection_method));
            }
        })
        .catch(error => {
            console.error('Error getting camera info:', error);
        });
}

// Update hardware status
function updateHardwareStatus() {
    fetch('/hardware_status')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Update camera status
                const cameraStatus = data.camera;
                const systemStatus = $('#systemStatus');
                const ledStatus = $('#ledStatus');

                if (cameraStatus.connected) {
                    if (cameraStatus.simulation) {
                        systemStatus.text('Simulation Mode');
                        systemStatus.addClass('text-warning');
                    } else {
                        systemStatus.text('Online');
                        systemStatus.removeClass('text-warning');
                    }
                } else {
                    systemStatus.text('Camera Offline');
                    systemStatus.addClass('text-danger');
                }

                // Update LED status
                if (data.led_controller.connected) {
                    ledStatus.text('Connected');
                    ledStatus.removeClass('text-danger');
                } else {
                    ledStatus.text('Disconnected');
                    ledStatus.addClass('text-danger');
                }
                
                // Update system info modal
                $('#cameraModel').text(cameraStatus.name || 'Unknown');
                $('#cameraResolution').text(data.camera.resolution || 'Unknown');
                $('#cameraFrameRate').text(data.camera.frame_rate ? `${data.camera.frame_rate} fps` : 'Unknown');
                $('#cameraMode').text(cameraStatus.simulation ? 'Simulation' : 'Live');
                
                $('#ledControllerStatus').text(data.led_controller.connected ? 'Connected' : 'Disconnected');
                $('#ledControllerPort').text(data.led_controller.port || 'Unknown');
                $('#ledBrightnessInfo').text(settings.led_brightness);
                $('#ledPatternInfo').text(settings.led_pattern);
            }
        })
        .catch(error => {
            console.error('Error checking hardware status:', error);
            // If we can't connect, update the status
            $('#systemStatus').text('Connection Error').addClass('text-danger');
        });
}

// Update the image counter
function updateCaptureCounter() {
    fetch('/image_count')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                $('#captureCounter').text(data.count);
                $('#imageCount').text(data.count);
            }
        })
        .catch(error => {
            console.error('Error getting image count:', error);
        });
}

// Show an alert to the user
function showAlert(message, type, duration = 5000) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;

    document.querySelector('.container').prepend(alertDiv);

    // Auto-dismiss after specified duration
    setTimeout(() => {
        // Bootstrap 5 way of removing the alert
        alertDiv.classList.remove('show');
        setTimeout(() => {
            alertDiv.remove();
        }, 150); // Wait for fade out animation
    }, duration);
}

// Utility function to capitalize first letter
function capitalizeFirstLetter(string) {
    return string.charAt(0).toUpperCase() + string.slice(1);
}

// Check for document visibility changes to handle tab switching
document.addEventListener('visibilitychange', function() {
    if (!document.hidden) {
        // When tab becomes visible again, refresh data
        updateHardwareStatus();
        updateCaptureCounter();
    }
});