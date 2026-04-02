package com.example.halidetest;

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

import androidx.appcompat.app.AppCompatActivity;

import java.io.File;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private Spinner spinnerOperation;
    private RadioGroup radioFramework;
    private Spinner spinnerResolution;
    private EditText editIterations;
    private Button btnRun;
    private Button btnRunAll;
    private TextView txtResult;
    private TextView txtVersionInfo;
    private ImageView imgResult;

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final Handler mainHandler = new Handler(Looper.getMainLooper());

    private String csvPath;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        spinnerOperation = findViewById(R.id.spinnerOperation);
        radioFramework = findViewById(R.id.radioFramework);
        spinnerResolution = findViewById(R.id.spinnerResolution);
        editIterations = findViewById(R.id.editIterations);
        btnRun = findViewById(R.id.btnRun);
        btnRunAll = findViewById(R.id.btnRunAll);
        txtResult = findViewById(R.id.txtResult);
        txtVersionInfo = findViewById(R.id.txtVersionInfo);
        imgResult = findViewById(R.id.imgResult);

        txtVersionInfo.setText("Halide 21.0.0 | OpenCV 3.4.16");

        // CSV file in app's external files directory
        File dir = getExternalFilesDir(null);
        if (dir != null) {
            csvPath = new File(dir, "benchmark_results.csv").getAbsolutePath();
        }

        btnRun.setOnClickListener(v -> runSingleBenchmark());
        btnRunAll.setOnClickListener(v -> runAllBenchmarks());
    }

    private int[] parseResolution(String res) {
        String[] parts = res.split("x");
        return new int[]{Integer.parseInt(parts[0]), Integer.parseInt(parts[1])};
    }

    private void runSingleBenchmark() {
        String operation = spinnerOperation.getSelectedItem().toString();
        String resolution = spinnerResolution.getSelectedItem().toString();
        int iterations = Integer.parseInt(editIterations.getText().toString());
        int[] dim = parseResolution(resolution);

        int checkedId = radioFramework.getCheckedRadioButtonId();
        boolean runHalide = (checkedId == R.id.radioHalide || checkedId == R.id.radioBoth);
        boolean runOpenCV = (checkedId == R.id.radioOpenCV || checkedId == R.id.radioBoth);

        btnRun.setEnabled(false);
        btnRunAll.setEnabled(false);
        txtResult.setText("Running...");

        executor.execute(() -> {
            StringBuilder sb = new StringBuilder();

            if (runHalide) {
                BenchmarkRunner.Result r = runOperation(operation, true, dim[0], dim[1], iterations);
                sb.append(r.toDisplayString()).append("\n\n");
                saveCsv(r);
            }
            if (runOpenCV) {
                BenchmarkRunner.Result r = runOperation(operation, false, dim[0], dim[1], iterations);
                sb.append(r.toDisplayString()).append("\n\n");
                saveCsv(r);
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

        btnRun.setEnabled(false);
        btnRunAll.setEnabled(false);
        txtResult.setText("Running all operations...");

        executor.execute(() -> {
            String[] operations = getResources().getStringArray(R.array.operations);
            StringBuilder sb = new StringBuilder();

            for (String op : operations) {
                // Halide
                mainHandler.post(() -> txtResult.setText("Running: " + op + " [Halide]..."));
                BenchmarkRunner.Result rH = runOperation(op, true, dim[0], dim[1], iterations);
                sb.append(rH.toDisplayString()).append("\n\n");
                saveCsv(rH);

                // OpenCV
                mainHandler.post(() -> txtResult.setText("Running: " + op + " [OpenCV]..."));
                BenchmarkRunner.Result rO = runOperation(op, false, dim[0], dim[1], iterations);
                sb.append(rO.toDisplayString()).append("\n\n");
                saveCsv(rO);

                // Speedup
                if (rH.median > 0) {
                    double speedup = (double) rO.median / rH.median;
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

    private BenchmarkRunner.Result runOperation(String opName, boolean useHalide,
                                                 int width, int height, int iterations) {
        // Create input bitmap with gradient pattern
        Bitmap inputBitmap = createTestBitmap(width, height);

        BenchmarkRunner.Operation op;

        switch (opName) {
            case "RGB to BGR": {
                Bitmap outputBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
                op = (halide) -> NativeBridge.rgbBgr(inputBitmap, outputBitmap, halide);
                break;
            }
            case "NV21 to RGB": {
                byte[] nv21 = createTestNv21(width, height);
                Bitmap outputBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
                op = (halide) -> NativeBridge.nv21ToRgb(nv21, width, height, outputBitmap, halide);
                break;
            }
            case "Gaussian Blur (5x5)": {
                Bitmap outputBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
                op = (halide) -> NativeBridge.gaussianBlur(inputBitmap, outputBitmap, 5, halide);
                break;
            }
            case "Lens Blur (r=4)": {
                Bitmap outputBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
                op = (halide) -> NativeBridge.lensBlur(inputBitmap, outputBitmap, 4, halide);
                break;
            }
            case "Resize Bilinear (0.5x)": {
                int ow = width / 2, oh = height / 2;
                Bitmap outputBitmap = Bitmap.createBitmap(ow, oh, Bitmap.Config.ARGB_8888);
                op = (halide) -> NativeBridge.resize(inputBitmap, outputBitmap, ow, oh, false, halide);
                break;
            }
            case "Resize Bicubic (0.5x)": {
                int ow = width / 2, oh = height / 2;
                Bitmap outputBitmap = Bitmap.createBitmap(ow, oh, Bitmap.Config.ARGB_8888);
                op = (halide) -> NativeBridge.resize(inputBitmap, outputBitmap, ow, oh, true, halide);
                break;
            }
            case "Rotate 90": {
                Bitmap outputBitmap = Bitmap.createBitmap(height, width, Bitmap.Config.ARGB_8888);
                op = (halide) -> NativeBridge.rotate(inputBitmap, outputBitmap, 90.0f, halide);
                break;
            }
            case "Rotate Arbitrary (45\u00B0)": {
                Bitmap outputBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
                op = (halide) -> NativeBridge.rotate(inputBitmap, outputBitmap, 45.0f, halide);
                break;
            }
            case "RGB to NV21": {
                int ySize = width * height;
                int uvSize = width * (height / 2);
                byte[] nv21Out = new byte[ySize + uvSize];
                op = (halide) -> NativeBridge.rgbToNv21(inputBitmap, nv21Out, halide);
                break;
            }
            case "Resize Area (0.5x)": {
                int ow = width / 2, oh = height / 2;
                Bitmap outputBitmap = Bitmap.createBitmap(ow, oh, Bitmap.Config.ARGB_8888);
                op = (halide) -> NativeBridge.resizeArea(inputBitmap, outputBitmap, ow, oh, halide);
                break;
            }
            case "Resize Letterbox (720p)": {
                int tw = 1280, th = 720;
                Bitmap outputBitmap = Bitmap.createBitmap(tw, th, Bitmap.Config.ARGB_8888);
                op = (halide) -> NativeBridge.resizeLetterbox(inputBitmap, outputBitmap, tw, th, halide);
                break;
            }
            case "Flip Horizontal": {
                Bitmap outputBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
                op = (halide) -> NativeBridge.flip(inputBitmap, outputBitmap, true, halide);
                break;
            }
            case "Flip Vertical": {
                Bitmap outputBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
                op = (halide) -> NativeBridge.flip(inputBitmap, outputBitmap, false, halide);
                break;
            }
            case "Resize Bilinear Target (720p)": {
                int tw = 1280, th = 720;
                Bitmap outputBitmap = Bitmap.createBitmap(tw, th, Bitmap.Config.ARGB_8888);
                op = (halide) -> NativeBridge.resizeBilinearTarget(inputBitmap, outputBitmap, tw, th, halide);
                break;
            }
            case "Resize Bicubic Target (720p)": {
                int tw = 1280, th = 720;
                Bitmap outputBitmap = Bitmap.createBitmap(tw, th, Bitmap.Config.ARGB_8888);
                op = (halide) -> NativeBridge.resizeBicubicTarget(inputBitmap, outputBitmap, tw, th, halide);
                break;
            }
            case "Resize Area Target (720p)": {
                int tw = 1280, th = 720;
                Bitmap outputBitmap = Bitmap.createBitmap(tw, th, Bitmap.Config.ARGB_8888);
                op = (halide) -> NativeBridge.resizeAreaTarget(inputBitmap, outputBitmap, tw, th, halide);
                break;
            }
            case "NV21 Pipeline Bilinear (rotate+resize)": {
                byte[] nv21 = createTestNv21(width, height);
                op = (halide) -> NativeBridge.nv21RotateResizeRgb(nv21, width, height,
                        90, 0, 1280, 720, false, halide);
                break;
            }
            case "NV21 Pipeline Area (rotate+resize)": {
                byte[] nv21 = createTestNv21(width, height);
                op = (halide) -> NativeBridge.nv21RotateResizeRgb(nv21, width, height,
                        90, 0, 1280, 720, true, halide);
                break;
            }
            case "NV21 YUV444 RGB (bilinear UV)": {
                byte[] nv21 = createTestNv21(width, height);
                Bitmap outputBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
                op = (halide) -> NativeBridge.nv21Yuv444Rgb(nv21, width, height, outputBitmap, halide);
                break;
            }
            case "NV21 to RGB Full-Range": {
                byte[] nv21 = createTestNv21(width, height);
                Bitmap outputBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
                op = (halide) -> NativeBridge.nv21ToRgbFullRange(nv21, width, height, outputBitmap, halide);
                break;
            }
            case "NV21 Resize+Pad+Rotate (384)": {
                byte[] nv21 = createTestNv21(width, height);
                int targetSize = 384;
                op = (halide) -> NativeBridge.nv21ResizePadRotate(nv21, width, height,
                        90, targetSize, halide);
                break;
            }
            case "Seg Argmax (8 classes)": {
                op = (halide) -> NativeBridge.segArgmax(width, height, 8, halide);
                break;
            }
            default: {
                Bitmap outputBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
                op = (halide) -> 0L;
                break;
            }
        }

        return BenchmarkRunner.run(opName, useHalide, width, height, iterations, op);
    }

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
        // NV21: Y plane (w*h) + interleaved VU plane (w*h/2)
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
