package com.example.halidetest;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.RadioGroup;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private Spinner spinnerOperation;
    private RadioGroup radioFramework;
    private Spinner spinnerResolution;
    private EditText editIterations;
    private EditText editTargetWidth;
    private EditText editTargetHeight;
    private Button btnRun;
    private Button btnRunAll;
    private Button btnCameraCapture;
    private Button btnCleanImages;
    private TextView txtResult;
    private TextView txtVersionInfo;
    private ImageView imgResult;

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final Handler mainHandler = new Handler(Looper.getMainLooper());

    private String csvPath;
    private String imageDir;

    // Camera capture state
    private byte[] capturedNv21Data;
    private int capturedWidth;
    private int capturedHeight;
    private boolean useCameraInput = false;

    private static final int CAMERA_PERMISSION_REQUEST = 100;
    private static final int CAMERA_ACTIVITY_REQUEST = 101;

    private static final String STATE_RESULT_TEXT = "result_text";
    private static final String STATE_USE_CAMERA = "use_camera";
    private static final String STATE_CAPTURED_W = "captured_w";
    private static final String STATE_CAPTURED_H = "captured_h";
    private static final String STATE_NV21_PATH = "nv21_state_path";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        spinnerOperation = findViewById(R.id.spinnerOperation);
        radioFramework = findViewById(R.id.radioFramework);
        spinnerResolution = findViewById(R.id.spinnerResolution);
        editIterations = findViewById(R.id.editIterations);
        editTargetWidth = findViewById(R.id.editTargetWidth);
        editTargetHeight = findViewById(R.id.editTargetHeight);
        btnRun = findViewById(R.id.btnRun);
        btnRunAll = findViewById(R.id.btnRunAll);
        btnCameraCapture = findViewById(R.id.btnCameraCapture);
        btnCleanImages = findViewById(R.id.btnCleanImages);
        txtResult = findViewById(R.id.txtResult);
        txtVersionInfo = findViewById(R.id.txtVersionInfo);
        imgResult = findViewById(R.id.imgResult);

        txtVersionInfo.setText("Halide 21.0.0 | OpenCV 3.4.16");

        // CSV file in app's external files directory
        File dir = getExternalFilesDir(null);
        if (dir != null) {
            csvPath = new File(dir, "benchmark_results.csv").getAbsolutePath();
            File imgDir = new File(dir, "benchmark_images");
            imgDir.mkdirs();
            imageDir = imgDir.getAbsolutePath();
        }

        btnRun.setOnClickListener(v -> runSingleBenchmark());
        btnRunAll.setOnClickListener(v -> runAllBenchmarks());
        btnCameraCapture.setOnClickListener(v -> requestCameraCapture());
        btnCleanImages.setOnClickListener(v -> cleanSavedImages());

        // Restore state after process death
        if (savedInstanceState != null) {
            String resultText = savedInstanceState.getString(STATE_RESULT_TEXT);
            if (resultText != null) {
                txtResult.setText(resultText);
            }

            useCameraInput = savedInstanceState.getBoolean(STATE_USE_CAMERA, false);
            capturedWidth = savedInstanceState.getInt(STATE_CAPTURED_W, 0);
            capturedHeight = savedInstanceState.getInt(STATE_CAPTURED_H, 0);

            String nv21StatePath = savedInstanceState.getString(STATE_NV21_PATH);
            if (useCameraInput && nv21StatePath != null) {
                File nv21File = new File(nv21StatePath);
                if (nv21File.exists()) {
                    byte[] nv21 = new byte[(int) nv21File.length()];
                    try (FileInputStream fis = new FileInputStream(nv21File)) {
                        fis.read(nv21);
                        capturedNv21Data = nv21;
                        // Restore preview
                        Bitmap preview = Bitmap.createBitmap(capturedWidth, capturedHeight,
                                Bitmap.Config.ARGB_8888);
                        NativeBridge.nv21ToRgb(capturedNv21Data, capturedWidth, capturedHeight,
                                preview, true);
                        imgResult.setImageBitmap(preview);
                    } catch (IOException | OutOfMemoryError ignored) {
                        useCameraInput = false;
                    }
                }
            }
        }
    }

    @Override
    protected void onSaveInstanceState(@NonNull Bundle outState) {
        super.onSaveInstanceState(outState);
        outState.putString(STATE_RESULT_TEXT, txtResult.getText().toString());
        outState.putBoolean(STATE_USE_CAMERA, useCameraInput);
        outState.putInt(STATE_CAPTURED_W, capturedWidth);
        outState.putInt(STATE_CAPTURED_H, capturedHeight);

        // Save NV21 data to temp file (too large for Bundle)
        if (useCameraInput && capturedNv21Data != null) {
            File nv21File = new File(getCacheDir(), "saved_nv21_state.bin");
            try (FileOutputStream fos = new FileOutputStream(nv21File)) {
                fos.write(capturedNv21Data);
                outState.putString(STATE_NV21_PATH, nv21File.getAbsolutePath());
            } catch (IOException ignored) {
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.shutdownNow();
    }

    private int[] parseResolution(String res) {
        String[] parts = res.split("x");
        return new int[]{Integer.parseInt(parts[0]), Integer.parseInt(parts[1])};
    }

    private int[] parseTargetDimensions() {
        try {
            int tw = Integer.parseInt(editTargetWidth.getText().toString().trim());
            int th = Integer.parseInt(editTargetHeight.getText().toString().trim());
            if (tw > 0 && th > 0) {
                return new int[]{tw, th};
            }
        } catch (NumberFormatException ignored) {
        }
        return new int[]{1280, 720};
    }

    // ──────────────────────────────────────────────────────────────
    // Camera Capture (via CameraActivity)
    // ──────────────────────────────────────────────────────────────

    private void requestCameraCapture() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST);
        } else {
            launchCameraActivity();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                            @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_REQUEST) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                launchCameraActivity();
            } else {
                Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private void launchCameraActivity() {
        Intent intent = new Intent(this, CameraActivity.class);
        startActivityForResult(intent, CAMERA_ACTIVITY_REQUEST);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == CAMERA_ACTIVITY_REQUEST && resultCode == RESULT_OK && data != null) {
            String nv21Path = data.getStringExtra(CameraActivity.EXTRA_NV21_PATH);
            int w = data.getIntExtra(CameraActivity.EXTRA_WIDTH, 0);
            int h = data.getIntExtra(CameraActivity.EXTRA_HEIGHT, 0);

            if (nv21Path != null && w > 0 && h > 0) {
                // Read NV21 data from temp file
                File file = new File(nv21Path);
                byte[] nv21 = new byte[(int) file.length()];
                try (FileInputStream fis = new FileInputStream(file)) {
                    fis.read(nv21);
                } catch (IOException e) {
                    txtResult.setText("Failed to read captured frame: " + e.getMessage());
                    return;
                }

                capturedNv21Data = nv21;
                capturedWidth = w;
                capturedHeight = h;
                useCameraInput = true;

                // Show preview
                try {
                    Bitmap preview = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
                    NativeBridge.nv21ToRgb(nv21, w, h, preview, true);
                    imgResult.setImageBitmap(preview);
                } catch (OutOfMemoryError ignored) {
                }

                txtResult.setText("Captured " + w + "x" + h +
                        " frame. Ready to benchmark.\n" +
                        "(Camera input mode active \u2014 Run to benchmark with captured frame)");

                // Clean up temp file
                file.delete();
            }
        }
    }

    // ──────────────────────────────────────────────────────────────
    // Benchmarking
    // ──────────────────────────────────────────────────────────────

    private void runSingleBenchmark() {
        String operation = spinnerOperation.getSelectedItem().toString();
        String resolution = spinnerResolution.getSelectedItem().toString();
        int iterations = Integer.parseInt(editIterations.getText().toString());
        int[] dim = parseResolution(resolution);
        int[] target = parseTargetDimensions();

        int checkedId = radioFramework.getCheckedRadioButtonId();
        boolean runHalide = (checkedId == R.id.radioHalide || checkedId == R.id.radioBoth);
        boolean runOpenCV = (checkedId == R.id.radioOpenCV || checkedId == R.id.radioBoth);

        btnRun.setEnabled(false);
        btnRunAll.setEnabled(false);
        txtResult.setText("Running...");

        executor.execute(() -> {
            StringBuilder sb = new StringBuilder();
            boolean inputSaved = false;

            if (runHalide) {
                OperationResult or = runOperation(operation, true, dim[0], dim[1],
                        iterations, target[0], target[1]);
                if (or == null) return; // OOM — error already posted
                sb.append(or.result.toDisplayString()).append("\n\n");
                saveCsv(or.result);
                if (useCameraInput && imageDir != null) {
                    if (!inputSaved) {
                        saveInputImage();
                        inputSaved = true;
                    }
                    saveOutputImage(or, operation, "Halide");
                }
            }
            if (runOpenCV) {
                OperationResult or = runOperation(operation, false, dim[0], dim[1],
                        iterations, target[0], target[1]);
                if (or == null) return;
                sb.append(or.result.toDisplayString()).append("\n\n");
                saveCsv(or.result);
                if (useCameraInput && imageDir != null) {
                    if (!inputSaved) {
                        saveInputImage();
                        inputSaved = true;
                    }
                    saveOutputImage(or, operation, "OpenCV");
                }
            }

            String result = sb.toString();
            mainHandler.post(() -> {
                txtResult.setText(result);
                btnRun.setEnabled(true);
                btnRunAll.setEnabled(true);
            });
        });
    }

    private void runAllBenchmarks() {
        int iterations = Integer.parseInt(editIterations.getText().toString());
        String resolution = spinnerResolution.getSelectedItem().toString();
        int[] dim = parseResolution(resolution);
        int[] target = parseTargetDimensions();

        btnRun.setEnabled(false);
        btnRunAll.setEnabled(false);
        txtResult.setText("Running all operations...");

        executor.execute(() -> {
            String[] operations = getResources().getStringArray(R.array.operations);
            StringBuilder sb = new StringBuilder();
            boolean inputSaved = false;

            for (String op : operations) {
                // Halide
                mainHandler.post(() -> txtResult.setText("Running: " + op + " [Halide]..."));
                OperationResult orH = runOperation(op, true, dim[0], dim[1],
                        iterations, target[0], target[1]);
                if (orH == null) continue;
                sb.append(orH.result.toDisplayString()).append("\n\n");
                saveCsv(orH.result);

                // OpenCV
                mainHandler.post(() -> txtResult.setText("Running: " + op + " [OpenCV]..."));
                OperationResult orO = runOperation(op, false, dim[0], dim[1],
                        iterations, target[0], target[1]);
                if (orO == null) continue;
                sb.append(orO.result.toDisplayString()).append("\n\n");
                saveCsv(orO.result);

                // Save images in camera mode
                if (useCameraInput && imageDir != null) {
                    if (!inputSaved) {
                        saveInputImage();
                        inputSaved = true;
                    }
                    saveOutputImage(orH, op, "Halide");
                    saveOutputImage(orO, op, "OpenCV");
                }

                // Speedup
                if (orH.result.median > 0) {
                    double speedup = (double) orO.result.median / orH.result.median;
                    sb.append(String.format("  >> Speedup: %.2fx\n\n", speedup));
                }
            }

            String result = sb.toString();
            mainHandler.post(() -> {
                txtResult.setText(result);
                btnRun.setEnabled(true);
                btnRunAll.setEnabled(true);
                Toast.makeText(this, "Results saved to: " + csvPath, Toast.LENGTH_LONG).show();
            });
        });
    }

    private static class OperationResult {
        BenchmarkRunner.Result result;
        Bitmap outputBitmap;   // non-null for bitmap-output operations
        byte[] outputNv21;     // non-null for NV21-output operations
        int outputWidth, outputHeight;
    }

    // Threshold: resolutions above 20MP use native-only path (no Java Bitmap)
    private static final long NATIVE_BENCHMARK_PIXEL_THRESHOLD = 20_000_000L;

    private int getOperationId(String opName) {
        String[] operations = getResources().getStringArray(R.array.operations);
        for (int i = 0; i < operations.length; i++) {
            if (operations[i].equals(opName)) return i;
        }
        return -1;
    }

    private OperationResult runNativeBenchmark(String opName, boolean useHalide,
                                                int width, int height, int iterations,
                                                int targetW, int targetH) {
        int opId = getOperationId(opName);
        if (opId < 0) return null;

        BenchmarkRunner.Operation op = (halide) ->
                NativeBridge.nativeBenchmark(opId, width, height, targetW, targetH, halide);

        OperationResult or = new OperationResult();
        or.outputWidth = targetW;
        or.outputHeight = targetH;
        or.result = BenchmarkRunner.run(opName, useHalide, width, height, iterations, op);
        return or;
    }

    private OperationResult runOperation(String opName, boolean useHalide,
                                          int width, int height, int iterations,
                                          int targetW, int targetH) {
        // When using camera input, override dimensions with captured frame size
        if (useCameraInput && capturedNv21Data != null) {
            width = capturedWidth;
            height = capturedHeight;
        }

        // For large resolutions, use native-only path (no Java Bitmap allocation)
        if ((long) width * height > NATIVE_BENCHMARK_PIXEL_THRESHOLD && !useCameraInput) {
            return runNativeBenchmark(opName, useHalide, width, height, iterations, targetW, targetH);
        }

        Bitmap inputBitmap;
        try {
            if (useCameraInput && capturedNv21Data != null) {
                inputBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
                NativeBridge.nv21ToRgb(capturedNv21Data, width, height, inputBitmap, true);
            } else {
                inputBitmap = createTestBitmap(width, height);
            }
        } catch (OutOfMemoryError e) {
            mainHandler.post(() -> {
                txtResult.setText("OOM: Resolution too large for this device.\nTry a smaller resolution or NV21-only operations.");
                btnRun.setEnabled(true);
                btnRunAll.setEnabled(true);
            });
            return null;
        }

        BenchmarkRunner.Operation op;
        OperationResult or = new OperationResult();
        or.outputWidth = width;
        or.outputHeight = height;

        final int w = width;
        final int h = height;

        switch (opName) {
            case "RGB BGR": {
                Bitmap outputBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
                op = (halide) -> NativeBridge.rgbBgrOptimized(inputBitmap, outputBitmap, halide);
                or.outputBitmap = outputBitmap;
                break;
            }
            case "NV21 to RGB": {
                byte[] nv21 = getNv21Input(w, h);
                Bitmap outputBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
                op = (halide) -> NativeBridge.nv21ToRgbOptimized(nv21, w, h, outputBitmap, halide);
                or.outputBitmap = outputBitmap;
                break;
            }
            case "RGB to NV21": {
                int ySize = w * h;
                int uvSize = w * (h / 2);
                byte[] nv21Out = new byte[ySize + uvSize];
                op = (halide) -> NativeBridge.rgbToNv21Optimized(inputBitmap, nv21Out, halide);
                or.outputNv21 = nv21Out;
                break;
            }
            case "NV21 YUV444 RGB (bilinear UV)": {
                byte[] nv21 = getNv21Input(w, h);
                Bitmap outputBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
                op = (halide) -> NativeBridge.nv21Yuv444Rgb(nv21, w, h, outputBitmap, halide);
                or.outputBitmap = outputBitmap;
                break;
            }
            case "NV21 to RGB Full-Range": {
                byte[] nv21 = getNv21Input(w, h);
                Bitmap outputBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
                op = (halide) -> NativeBridge.nv21ToRgbFullRange(nv21, w, h, outputBitmap, halide);
                or.outputBitmap = outputBitmap;
                break;
            }
            case "Gaussian Blur (5x5)": {
                Bitmap outputBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
                op = (halide) -> NativeBridge.gaussianBlur(inputBitmap, outputBitmap, 5, halide);
                or.outputBitmap = outputBitmap;
                break;
            }
            case "Lens Blur (r=4)": {
                Bitmap outputBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
                op = (halide) -> NativeBridge.lensBlur(inputBitmap, outputBitmap, 4, halide);
                or.outputBitmap = outputBitmap;
                break;
            }
            case "Resize Bilinear": {
                Bitmap outputBitmap = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888);
                op = (halide) -> NativeBridge.resizeBilinearOptimized(inputBitmap, outputBitmap, targetW, targetH, halide);
                or.outputBitmap = outputBitmap;
                or.outputWidth = targetW;
                or.outputHeight = targetH;
                break;
            }
            case "Resize Bicubic": {
                Bitmap outputBitmap = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888);
                op = (halide) -> NativeBridge.resizeBicubicOptimized(inputBitmap, outputBitmap, targetW, targetH, halide);
                or.outputBitmap = outputBitmap;
                or.outputWidth = targetW;
                or.outputHeight = targetH;
                break;
            }
            case "Resize Area": {
                Bitmap outputBitmap = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888);
                op = (halide) -> NativeBridge.resizeAreaOptimized(inputBitmap, outputBitmap, targetW, targetH, halide);
                or.outputBitmap = outputBitmap;
                or.outputWidth = targetW;
                or.outputHeight = targetH;
                break;
            }
            case "Resize Letterbox (720p)": {
                Bitmap outputBitmap = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888);
                op = (halide) -> NativeBridge.resizeLetterbox(inputBitmap, outputBitmap, targetW, targetH, halide);
                or.outputBitmap = outputBitmap;
                or.outputWidth = targetW;
                or.outputHeight = targetH;
                break;
            }
            case "Rotate 90": {
                Bitmap outputBitmap = Bitmap.createBitmap(h, w, Bitmap.Config.ARGB_8888);
                op = (halide) -> NativeBridge.rotate(inputBitmap, outputBitmap, 90.0f, halide);
                or.outputBitmap = outputBitmap;
                or.outputWidth = h;
                or.outputHeight = w;
                break;
            }
            case "Rotate Arbitrary (45\u00B0)": {
                Bitmap outputBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
                op = (halide) -> NativeBridge.rotate(inputBitmap, outputBitmap, 45.0f, halide);
                or.outputBitmap = outputBitmap;
                break;
            }
            case "Flip": {
                Bitmap outputBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
                op = (halide) -> NativeBridge.flip(inputBitmap, outputBitmap, true, halide);
                or.outputBitmap = outputBitmap;
                break;
            }
            case "NV21 Pipeline (rotate+resize)": {
                byte[] nv21 = getNv21Input(w, h);
                op = (halide) -> NativeBridge.nv21RotateResizeRgb(nv21, w, h,
                        90, 0, targetW, targetH, false, halide);
                or.outputWidth = targetW;
                or.outputHeight = targetH;
                break;
            }
            case "NV21 Resize+Pad+Rotate (384)": {
                byte[] nv21 = getNv21Input(w, h);
                int targetSize = 384;
                op = (halide) -> NativeBridge.nv21ResizePadRotate(nv21, w, h,
                        90, targetSize, halide);
                or.outputWidth = targetSize;
                or.outputHeight = targetSize;
                break;
            }
            case "NV21 Resize (stay NV21)": {
                byte[] nv21 = getNv21Input(w, h);
                op = (halide) -> NativeBridge.nv21ResizeBilinearOptimized(nv21, w, h, targetW, targetH, halide);
                or.outputWidth = targetW;
                or.outputHeight = targetH;
                break;
            }
            case "NV21 Resize+RGB (fused)": {
                byte[] nv21 = getNv21Input(w, h);
                op = (halide) -> NativeBridge.nv21ResizeRgbBilinearOptimized(nv21, w, h, targetW, targetH, halide);
                or.outputWidth = targetW;
                or.outputHeight = targetH;
                break;
            }
            case "Seg Argmax (8 classes)": {
                op = (halide) -> NativeBridge.segArgmax(w, h, 8, halide);
                break;
            }
            case "Seg Portrait Blur (r=8)": {
                Bitmap outputBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
                op = (halide) -> NativeBridge.segPortraitBlur(inputBitmap, outputBitmap, 8, halide);
                or.outputBitmap = outputBitmap;
                break;
            }
            case "Seg Background Replace": {
                Bitmap outputBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
                Bitmap bgBitmap = createTestBitmap(w, h);
                op = (halide) -> NativeBridge.segBgReplace(inputBitmap, bgBitmap, outputBitmap, halide);
                or.outputBitmap = outputBitmap;
                break;
            }
            case "Seg Color Style": {
                Bitmap outputBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
                op = (halide) -> NativeBridge.segColorStyle(inputBitmap, outputBitmap, halide);
                or.outputBitmap = outputBitmap;
                break;
            }
            case "Seg Depth Blur (3 zones)": {
                Bitmap outputBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
                op = (halide) -> NativeBridge.segDepthBlur(inputBitmap, outputBitmap, 3, halide);
                or.outputBitmap = outputBitmap;
                break;
            }
            default: {
                Bitmap outputBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
                op = (halide) -> 0L;
                or.outputBitmap = outputBitmap;
                break;
            }
        }

        or.result = BenchmarkRunner.run(opName, useHalide, w, h, iterations, op);
        return or;
    }

    private byte[] getNv21Input(int width, int height) {
        if (useCameraInput && capturedNv21Data != null) {
            return capturedNv21Data;
        }
        return createTestNv21(width, height);
    }

    // ──────────────────────────────────────────────────────────────
    // Image Saving
    // ──────────────────────────────────────────────────────────────

    private void saveInputImage() {
        if (capturedNv21Data == null || imageDir == null) return;
        try {
            Bitmap bmp = Bitmap.createBitmap(capturedWidth, capturedHeight, Bitmap.Config.ARGB_8888);
            NativeBridge.nv21ToRgb(capturedNv21Data, capturedWidth, capturedHeight, bmp, true);
            String filename = "input_" + capturedWidth + "x" + capturedHeight + ".png";
            saveBitmapToFile(bmp, filename);
            bmp.recycle();
        } catch (OutOfMemoryError ignored) {
        }
    }

    private void saveOutputImage(OperationResult or, String opName, String framework) {
        if (or == null || imageDir == null) return;
        String safeName = opName.replaceAll("[^a-zA-Z0-9]", "_");
        String filename = "output_" + safeName + "_" + framework + "_" +
                or.outputWidth + "x" + or.outputHeight + ".png";

        if (or.outputBitmap != null) {
            saveBitmapToFile(or.outputBitmap, filename);
        } else if (or.outputNv21 != null) {
            try {
                Bitmap bmp = Bitmap.createBitmap(or.outputWidth, or.outputHeight,
                        Bitmap.Config.ARGB_8888);
                NativeBridge.nv21ToRgb(or.outputNv21, or.outputWidth, or.outputHeight, bmp, true);
                saveBitmapToFile(bmp, filename);
                bmp.recycle();
            } catch (OutOfMemoryError ignored) {
            }
        }
    }

    private void saveBitmapToFile(Bitmap bitmap, String filename) {
        File file = new File(imageDir, filename);
        try (FileOutputStream fos = new FileOutputStream(file)) {
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, fos);
        } catch (IOException ignored) {
        }
    }

    // ──────────────────────────────────────────────────────────────
    // Clean Saved Images
    // ──────────────────────────────────────────────────────────────

    private void cleanSavedImages() {
        if (imageDir == null) return;
        File dir = new File(imageDir);
        if (dir.exists()) {
            File[] files = dir.listFiles();
            if (files != null) {
                int count = 0;
                for (File f : files) {
                    if (f.delete()) count++;
                }
                Toast.makeText(this, "Deleted " + count + " images", Toast.LENGTH_SHORT).show();
            }
        }
    }

    // ──────────────────────────────────────────────────────────────
    // Test Data Generation
    // ──────────────────────────────────────────────────────────────

    private Bitmap createTestBitmap(int width, int height) {
        Bitmap bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        int[] pixels = new int[width * height];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int r = (x * 255) / Math.max(width - 1, 1);
                int g = (y * 255) / Math.max(height - 1, 1);
                int b = ((x + y) * 255) / Math.max(width + height - 2, 1);
                pixels[y * width + x] = Color.argb(255, r, g, b);
            }
        }
        bmp.setPixels(pixels, 0, width, 0, 0, width, height);
        return bmp;
    }

    private byte[] createTestNv21(int width, int height) {
        int ySize = width * height;
        int uvSize = width * (height / 2);
        byte[] nv21 = new byte[ySize + uvSize];

        Random rand = new Random(42);
        for (int i = 0; i < ySize; i++) {
            nv21[i] = (byte) ((i % 256));
        }
        for (int i = 0; i < uvSize; i++) {
            nv21[ySize + i] = (byte) (128 + (i % 64));
        }
        return nv21;
    }

    private void saveCsv(BenchmarkRunner.Result r) {
        if (csvPath != null) {
            NativeBridge.appendCsv(csvPath, r.toCsvLine());
        }
    }
}
