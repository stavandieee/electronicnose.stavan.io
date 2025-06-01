/**
 * E-Nose ESP32 Firmware
 * Main firmware for AI-powered electronic nose system
 */

#include <Arduino.h>
#include <WiFi.h>
#include <ArduinoJson.h>

// Pin definitions for MQ sensors
#define MQ2_PIN 34
#define MQ3_PIN 35
#define MQ4_PIN 32
#define MQ5_PIN 33
#define MQ7_PIN 25
#define MQ135_PIN 26

// Sensor array
const int SENSOR_PINS[] = {MQ2_PIN, MQ3_PIN, MQ4_PIN, MQ5_PIN, MQ7_PIN, MQ135_PIN};
const int NUM_SENSORS = 6;
const char* SENSOR_NAMES[] = {"MQ2", "MQ3", "MQ4", "MQ5", "MQ7", "MQ135"};

// Sampling configuration
const int SAMPLES_PER_READING = 10;
const int SAMPLE_DELAY_MS = 50;

// Calibration values (R0 for each sensor in clean air)
float R0_VALUES[NUM_SENSORS] = {10.0, 10.0, 10.0, 10.0, 10.0, 10.0};

// TensorFlow Lite model placeholder
// In real implementation, include the quantized model here
// #include "model_data.h"

void setup() {
    Serial.begin(115200);
    Serial.println("E-Nose System Starting...");
    
    // Initialize sensor pins
    for (int i = 0; i < NUM_SENSORS; i++) {
        pinMode(SENSOR_PINS[i], INPUT);
    }
    
    // Allow sensors to warm up
    Serial.println("Warming up sensors (30 seconds)...");
    delay(30000);
    
    // Calibrate sensors in clean air
    calibrateSensors();
    
    Serial.println("E-Nose Ready!");
}

void loop() {
    // Read all sensors
    float sensorValues[NUM_SENSORS];
    readSensors(sensorValues);
    
    // Extract features
    float features[NUM_SENSORS * 3]; // 3 features per sensor
    extractFeatures(sensorValues, features);
    
    // Run inference (placeholder)
    int classification = runInference(features);
    
    // Output results
    outputResults(sensorValues, classification);
    
    delay(1000); // Read every second
}

void calibrateSensors() {
    Serial.println("Calibrating sensors in clean air...");
    
    for (int i = 0; i < NUM_SENSORS; i++) {
        float sum = 0;
        for (int j = 0; j < 100; j++) {
            sum += analogRead(SENSOR_PINS[i]);
            delay(10);
        }
        float avgValue = sum / 100.0;
        
        // Calculate R0 (sensor resistance in clean air)
        float sensorVolt = avgValue / 4095.0 * 3.3;
        float RS = (3.3 - sensorVolt) / sensorVolt;
        R0_VALUES[i] = RS;
        
        Serial.print(SENSOR_NAMES[i]);
        Serial.print(" calibrated. R0 = ");
        Serial.println(R0_VALUES[i]);
    }
}

void readSensors(float* values) {
    for (int i = 0; i < NUM_SENSORS; i++) {
        float sum = 0;
        
        // Take multiple samples and average
        for (int j = 0; j < SAMPLES_PER_READING; j++) {
            sum += analogRead(SENSOR_PINS[i]);
            delay(SAMPLE_DELAY_MS);
        }
        
        values[i] = sum / SAMPLES_PER_READING;
    }
}

void extractFeatures(float* rawValues, float* features) {
    for (int i = 0; i < NUM_SENSORS; i++) {
        // Convert to voltage
        float voltage = rawValues[i] / 4095.0 * 3.3;
        
        // Calculate sensor resistance
        float RS = (3.3 - voltage) / voltage;
        
        // Calculate RS/R0 ratio (main feature)
        float ratio = RS / R0_VALUES[i];
        
        // Feature 1: RS/R0 ratio
        features[i * 3] = ratio;
        
        // Feature 2: Normalized voltage
        features[i * 3 + 1] = voltage / 3.3;
        
        // Feature 3: Rate of change (simplified - would need history)
        features[i * 3 + 2] = 0.0; // Placeholder
    }
}

int runInference(float* features) {
    // Placeholder for TensorFlow Lite inference
    // In real implementation:
    // 1. Load the quantized model
    // 2. Set input tensor with features
    // 3. Run inference
    // 4. Get output predictions
    
    // For now, return a dummy classification
    return 0; // 0 = Fresh, 1 = Spoiled, etc.
}

void outputResults(float* sensorValues, int classification) {
    // Create JSON output
    StaticJsonDocument<512> doc;
    
    // Add timestamp
    doc["timestamp"] = millis();
    
    // Add raw sensor values
    JsonArray sensors = doc.createNestedArray("sensors");
    for (int i = 0; i < NUM_SENSORS; i++) {
        JsonObject sensor = sensors.createNestedObject();
        sensor["name"] = SENSOR_NAMES[i];
        sensor["value"] = sensorValues[i];
        sensor["voltage"] = sensorValues[i] / 4095.0 * 3.3;
    }
    
    // Add classification result
    const char* classes[] = {"Fresh", "Slightly Spoiled", "Spoiled", "Fermented", "Contaminated"};
    doc["classification"] = classes[classification];
    doc["confidence"] = 0.943; // Placeholder confidence
    
    // Send JSON over serial
    serializeJson(doc, Serial);
    Serial.println();
}