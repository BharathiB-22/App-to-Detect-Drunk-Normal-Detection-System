package com.example.apptodetect

import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions

class MainActivity : AppCompatActivity() {

    private lateinit var imagePreview: ImageView
    private lateinit var btnTakePicture: Button
    private lateinit var btnUploadImage: Button
    private lateinit var btnAnalyze: Button
    private lateinit var tvResult: TextView
    private lateinit var tvConfidence: TextView
    private lateinit var tvDetails: TextView

    private var selectedImageUri: Uri? = null
    private var selectedBitmap: Bitmap? = null

    private val cameraLauncher =
        registerForActivityResult(ActivityResultContracts.TakePicturePreview()) { bitmap: Bitmap? ->
            if (bitmap != null) {
                selectedBitmap = bitmap
                selectedImageUri = null
                imagePreview.setImageBitmap(bitmap)
                tvResult.text = "Result: Not analyzed yet"
                tvConfidence.text = "Confidence: --"
                tvDetails.text = "Details: Camera image captured and ready for analysis"
            } else {
                Toast.makeText(this, "Camera capture cancelled", Toast.LENGTH_SHORT).show()
            }
        }

    private val galleryLauncher =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            if (uri != null) {
                selectedImageUri = uri
                selectedBitmap = loadBitmapFromUri(uri)
                if (selectedBitmap != null) {
                    imagePreview.setImageBitmap(selectedBitmap)
                    tvResult.text = "Result: Not analyzed yet"
                    tvConfidence.text = "Confidence: --"
                    tvDetails.text = "Details: Image loaded and ready for analysis"
                } else {
                    Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show()
                }
            } else {
                Toast.makeText(this, "No image selected", Toast.LENGTH_SHORT).show()
            }
        }

    private val requestCameraPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                cameraLauncher.launch(null)
            } else {
                Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imagePreview = findViewById(R.id.imagePreview)
        btnTakePicture = findViewById(R.id.btnTakePicture)
        btnUploadImage = findViewById(R.id.btnUploadImage)
        btnAnalyze = findViewById(R.id.btnAnalyze)
        tvResult = findViewById(R.id.tvResult)
        tvConfidence = findViewById(R.id.tvConfidence)
        tvDetails = findViewById(R.id.tvDetails)

        btnTakePicture.setOnClickListener {
            when {
                checkSelfPermission(android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED -> {
                    cameraLauncher.launch(null)
                }
                else -> {
                    requestCameraPermission.launch(android.Manifest.permission.CAMERA)
                }
            }
        }

        btnUploadImage.setOnClickListener {
            galleryLauncher.launch("image/*")
        }

        btnAnalyze.setOnClickListener {
            if (selectedBitmap == null) {
                Toast.makeText(this, "Please upload or capture an image first", Toast.LENGTH_SHORT).show()
            } else {
                analyzeFace(selectedBitmap!!)
            }
        }
    }

    private fun loadBitmapFromUri(uri: Uri): Bitmap? {
        return try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                val source = ImageDecoder.createSource(contentResolver, uri)
                ImageDecoder.decodeBitmap(source)
            } else {
                MediaStore.Images.Media.getBitmap(contentResolver, uri)
            }
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    private fun analyzeFace(bitmap: Bitmap) {
        tvResult.text = "Result: Analyzing..."
        tvConfidence.text = "Confidence: --"
        tvDetails.text = "Details: Detecting face..."

        val image = InputImage.fromBitmap(bitmap, 0)

        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .build()

        val detector = FaceDetection.getClient(options)

        detector.process(image)
            .addOnSuccessListener { faces ->
                if (faces.isEmpty()) {
                    tvResult.text = "Result: No Face Found"
                    tvConfidence.text = "Confidence: --"
                    tvDetails.text = "Details: Please choose a clearer image"
                    return@addOnSuccessListener
                }

                val face = faces[0]

                val leftEye = face.leftEyeOpenProbability ?: -1f
                val rightEye = face.rightEyeOpenProbability ?: -1f
                val headTilt = face.headEulerAngleZ

                val avgEye =
                    if (leftEye >= 0f && rightEye >= 0f) (leftEye + rightEye) / 2f else -1f

                val eyeDiff =
                    if (leftEye >= 0f && rightEye >= 0f) kotlin.math.abs(leftEye - rightEye) else 0f

                if (avgEye > 0f) {
                    val (label, confidence, reasonText) = classifyFace(avgEye, headTilt, eyeDiff)

                    tvResult.text = "Result: $label"
                    tvConfidence.text = "Confidence: ${(confidence * 100).toInt()}%"
                    tvDetails.text = "Details: $reasonText"

                    if (label == "Drunk") {
                        tvResult.setTextColor(resources.getColor(android.R.color.holo_red_dark))
                    } else {
                        tvResult.setTextColor(resources.getColor(android.R.color.holo_green_dark))
                    }
                } else {
                    tvResult.text = "Result: Unable to Analyze"
                    tvConfidence.text = "Confidence: --"
                    tvDetails.text = "Details: Could not compute eye features"
                }
            }
            .addOnFailureListener { e ->
                tvResult.text = "Result: Detection Failed"
                tvConfidence.text = "Confidence: --"
                tvDetails.text = "Error: ${e.message}"
            }
    }

    private fun classifyFace(avgEye: Float, headTilt: Float, eyeDiff: Float): Triple<String, Float, String> {

        var score = 0
        val reasons = mutableListOf<String>()

        if (avgEye < 0.20f) {
            score += 40
            reasons.add("Eyes mostly closed")
        } else if (avgEye < 0.30f) {
            score += 25
            reasons.add("Eyes partially closed")
        }

        val tilt = kotlin.math.abs(headTilt)
        if (tilt > 10f) {
            score += 30
            reasons.add("Head tilted strongly")
        } else if (tilt > 5f) {
            score += 15
            reasons.add("Head slightly tilted")
        }

        if (eyeDiff > 0.1f) {
            score += 10
            reasons.add("Uneven eye openness")
        }

        val label = if (score >= 40) "Drunk" else "Normal"
        val confidence = (score / 100f).coerceAtMost(1f)

        return Triple(label, confidence, reasons.joinToString(", "))
    }
}