package com.example.halidetest;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.ImageFormat;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import android.util.Size;
import android.view.Surface;
import android.view.TextureView;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;

public class CameraActivity extends AppCompatActivity {

    public static final String EXTRA_NV21_PATH = "nv21_path";
    public static final String EXTRA_WIDTH = "width";
    public static final String EXTRA_HEIGHT = "height";

    private TextureView texturePreview;
    private Button btnCapture;
    private TextView txtCameraInfo;

    private CameraDevice cameraDevice;
    private CameraCaptureSession captureSession;
    private ImageReader imageReader;
    private HandlerThread cameraThread;
    private Handler cameraHandler;
    private Handler mainHandler = new Handler(Looper.getMainLooper());

    private String cameraId;
    private Size previewSize;
    private Size captureSize;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);

        texturePreview = findViewById(R.id.texturePreview);
        btnCapture = findViewById(R.id.btnCapture);
        txtCameraInfo = findViewById(R.id.txtCameraInfo);

        btnCapture.setOnClickListener(v -> captureFrame());

        texturePreview.setSurfaceTextureListener(new TextureView.SurfaceTextureListener() {
            @Override
            public void onSurfaceTextureAvailable(@NonNull SurfaceTexture surface,
                                                   int width, int height) {
                startCameraThread();
                setupCamera();
                openCamera();
            }

            @Override
            public void onSurfaceTextureSizeChanged(@NonNull SurfaceTexture surface,
                                                     int width, int height) {
            }

            @Override
            public boolean onSurfaceTextureDestroyed(@NonNull SurfaceTexture surface) {
                return true;
            }

            @Override
            public void onSurfaceTextureUpdated(@NonNull SurfaceTexture surface) {
            }
        });
    }

    @Override
    protected void onPause() {
        super.onPause();
        closeCamera();
        stopCameraThread();
    }

    private void startCameraThread() {
        cameraThread = new HandlerThread("CameraThread");
        cameraThread.start();
        cameraHandler = new Handler(cameraThread.getLooper());
    }

    private void stopCameraThread() {
        if (cameraThread != null) {
            cameraThread.quitSafely();
            try {
                cameraThread.join();
            } catch (InterruptedException ignored) {
            }
            cameraThread = null;
            cameraHandler = null;
        }
    }

    private void setupCamera() {
        CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        if (manager == null) return;

        try {
            for (String id : manager.getCameraIdList()) {
                CameraCharacteristics chars = manager.getCameraCharacteristics(id);
                Integer facing = chars.get(CameraCharacteristics.LENS_FACING);
                if (facing == null || facing != CameraCharacteristics.LENS_FACING_BACK) continue;

                cameraId = id;
                StreamConfigurationMap map = chars.get(
                        CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
                if (map == null) continue;

                // Pick largest YUV size for capture
                Size[] yuvSizes = map.getOutputSizes(ImageFormat.YUV_420_888);
                captureSize = yuvSizes[0];
                for (Size s : yuvSizes) {
                    if ((long) s.getWidth() * s.getHeight() >
                            (long) captureSize.getWidth() * captureSize.getHeight()) {
                        captureSize = s;
                    }
                }

                // Pick a preview size that fits the TextureView aspect ratio
                Size[] previewSizes = map.getOutputSizes(SurfaceTexture.class);
                previewSize = choosePreviewSize(previewSizes,
                        texturePreview.getWidth(), texturePreview.getHeight());

                mainHandler.post(() ->
                        txtCameraInfo.setText("Capture: " + captureSize.getWidth() + "x" +
                                captureSize.getHeight() + "  |  Preview: " +
                                previewSize.getWidth() + "x" + previewSize.getHeight()));
                break;
            }
        } catch (CameraAccessException e) {
            Toast.makeText(this, "Camera setup failed: " + e.getMessage(),
                    Toast.LENGTH_SHORT).show();
        }
    }

    private Size choosePreviewSize(Size[] sizes, int viewWidth, int viewHeight) {
        // Prefer 1080p or closest size that fits
        Size best = sizes[0];
        for (Size s : sizes) {
            if (s.getWidth() == 1920 && s.getHeight() == 1080) return s;
            if (s.getWidth() <= 1920 && s.getHeight() <= 1080 &&
                    (long) s.getWidth() * s.getHeight() >
                            (long) best.getWidth() * best.getHeight()) {
                best = s;
            }
        }
        return best;
    }

    private void openCamera() {
        if (cameraId == null) return;
        CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        if (manager == null) return;

        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            finish();
            return;
        }

        try {
            manager.openCamera(cameraId, new CameraDevice.StateCallback() {
                @Override
                public void onOpened(@NonNull CameraDevice camera) {
                    cameraDevice = camera;
                    startPreview();
                }

                @Override
                public void onDisconnected(@NonNull CameraDevice camera) {
                    camera.close();
                    cameraDevice = null;
                }

                @Override
                public void onError(@NonNull CameraDevice camera, int error) {
                    camera.close();
                    cameraDevice = null;
                    mainHandler.post(() -> {
                        Toast.makeText(CameraActivity.this,
                                "Camera error: " + error, Toast.LENGTH_SHORT).show();
                        finish();
                    });
                }
            }, cameraHandler);
        } catch (CameraAccessException e) {
            Toast.makeText(this, "Cannot open camera: " + e.getMessage(),
                    Toast.LENGTH_SHORT).show();
        }
    }

    private void startPreview() {
        if (cameraDevice == null || !texturePreview.isAvailable()) return;

        try {
            SurfaceTexture surfaceTexture = texturePreview.getSurfaceTexture();
            surfaceTexture.setDefaultBufferSize(previewSize.getWidth(), previewSize.getHeight());
            Surface previewSurface = new Surface(surfaceTexture);

            // Create ImageReader for capture
            imageReader = ImageReader.newInstance(
                    captureSize.getWidth(), captureSize.getHeight(),
                    ImageFormat.YUV_420_888, 2);

            CaptureRequest.Builder previewBuilder =
                    cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            previewBuilder.addTarget(previewSurface);
            previewBuilder.set(CaptureRequest.CONTROL_AF_MODE,
                    CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);

            cameraDevice.createCaptureSession(
                    Arrays.asList(previewSurface, imageReader.getSurface()),
                    new CameraCaptureSession.StateCallback() {
                        @Override
                        public void onConfigured(@NonNull CameraCaptureSession session) {
                            captureSession = session;
                            try {
                                session.setRepeatingRequest(previewBuilder.build(), null,
                                        cameraHandler);
                                mainHandler.post(() -> btnCapture.setEnabled(true));
                            } catch (CameraAccessException e) {
                                mainHandler.post(() ->
                                        txtCameraInfo.setText("Preview failed: " + e.getMessage()));
                            }
                        }

                        @Override
                        public void onConfigureFailed(@NonNull CameraCaptureSession session) {
                            mainHandler.post(() ->
                                    txtCameraInfo.setText("Session configuration failed"));
                        }
                    }, cameraHandler);

        } catch (CameraAccessException e) {
            txtCameraInfo.setText("Preview setup failed: " + e.getMessage());
        }
    }

    private void captureFrame() {
        if (cameraDevice == null || captureSession == null || imageReader == null) return;

        btnCapture.setEnabled(false);
        txtCameraInfo.setText("Capturing...");

        imageReader.setOnImageAvailableListener(reader -> {
            Image image = reader.acquireLatestImage();
            if (image == null) return;

            byte[] nv21 = yuv420ToNv21(image);
            int w = image.getWidth();
            int h = image.getHeight();
            image.close();

            // Save NV21 to temp file (too large for Intent extras)
            File tempFile = new File(getCacheDir(), "captured_nv21.raw");
            try (FileOutputStream fos = new FileOutputStream(tempFile)) {
                fos.write(nv21);
            } catch (IOException e) {
                mainHandler.post(() -> {
                    txtCameraInfo.setText("Failed to save capture: " + e.getMessage());
                    btnCapture.setEnabled(true);
                });
                return;
            }

            mainHandler.post(() -> {
                Intent resultIntent = new Intent();
                resultIntent.putExtra(EXTRA_NV21_PATH, tempFile.getAbsolutePath());
                resultIntent.putExtra(EXTRA_WIDTH, w);
                resultIntent.putExtra(EXTRA_HEIGHT, h);
                setResult(RESULT_OK, resultIntent);
                finish();
            });
        }, cameraHandler);

        try {
            CaptureRequest.Builder captureBuilder =
                    cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE);
            captureBuilder.addTarget(imageReader.getSurface());
            captureBuilder.set(CaptureRequest.CONTROL_AF_MODE,
                    CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);

            captureSession.capture(captureBuilder.build(),
                    new CameraCaptureSession.CaptureCallback() {
                        @Override
                        public void onCaptureCompleted(@NonNull CameraCaptureSession session,
                                                       @NonNull CaptureRequest request,
                                                       @NonNull TotalCaptureResult result) {
                            // Image will be delivered to ImageReader listener
                        }
                    }, cameraHandler);
        } catch (CameraAccessException e) {
            txtCameraInfo.setText("Capture failed: " + e.getMessage());
            btnCapture.setEnabled(true);
        }
    }

    private void closeCamera() {
        if (captureSession != null) {
            captureSession.close();
            captureSession = null;
        }
        if (cameraDevice != null) {
            cameraDevice.close();
            cameraDevice = null;
        }
        if (imageReader != null) {
            imageReader.close();
            imageReader = null;
        }
    }

    private byte[] yuv420ToNv21(Image image) {
        int width = image.getWidth();
        int height = image.getHeight();
        int ySize = width * height;
        int uvSize = width * (height / 2);
        byte[] nv21 = new byte[ySize + uvSize];

        Image.Plane yPlane = image.getPlanes()[0];
        Image.Plane uPlane = image.getPlanes()[1];
        Image.Plane vPlane = image.getPlanes()[2];

        // Copy Y plane
        ByteBuffer yBuffer = yPlane.getBuffer();
        int yRowStride = yPlane.getRowStride();
        if (yRowStride == width) {
            yBuffer.get(nv21, 0, ySize);
        } else {
            for (int row = 0; row < height; row++) {
                yBuffer.position(row * yRowStride);
                yBuffer.get(nv21, row * width, width);
            }
        }

        // Copy UV planes into interleaved VU (NV21 format)
        int vPixelStride = vPlane.getPixelStride();
        int vRowStride = vPlane.getRowStride();

        if (vPixelStride == 2) {
            // Most devices: V and U buffers are already interleaved as VUVU...
            ByteBuffer vBuffer = vPlane.getBuffer();
            int uvHeight = height / 2;
            if (vRowStride == width) {
                vBuffer.get(nv21, ySize, Math.min(vBuffer.remaining(), uvSize));
            } else {
                for (int row = 0; row < uvHeight; row++) {
                    vBuffer.position(row * vRowStride);
                    vBuffer.get(nv21, ySize + row * width, Math.min(width, vBuffer.remaining()));
                }
            }
        } else {
            // Pixel stride == 1: manually interleave V and U
            ByteBuffer uBuffer = uPlane.getBuffer();
            ByteBuffer vBuffer = vPlane.getBuffer();
            int uRowStride = uPlane.getRowStride();
            int uvHeight = height / 2;
            int uvWidth = width / 2;

            for (int row = 0; row < uvHeight; row++) {
                for (int col = 0; col < uvWidth; col++) {
                    int vIdx = row * vRowStride + col;
                    int uIdx = row * uRowStride + col;
                    int outIdx = ySize + row * width + col * 2;
                    nv21[outIdx] = vBuffer.get(vIdx);
                    nv21[outIdx + 1] = uBuffer.get(uIdx);
                }
            }
        }

        return nv21;
    }
}
