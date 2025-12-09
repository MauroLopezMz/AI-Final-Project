#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_error_reporter.h" // Added for debugging!

// Include your trained brain
#include "model_data_32.h"

// --- PIN DEFINITIONS ---
const int trigPinL = 13; const int echoPinL = 12;
const int trigPinF = 14; const int echoPinF = 27;
const int trigPinR = 26; const int echoPinR = 25;

const int motorL1 = 33; const int motorL2 = 32;
const int motorR1 = 18; const int motorR2 = 19;

const int irSensorPin = 23;

// --- HYPERPARAMETERS ---
const float MAX_SENSOR_RANGE = 1.5; 

// --- TFLite GLOBALS ---
// INCREASED MEMORY TO 60KB TO PREVENT CRASHES
const int kTensorArenaSize = 60 * 1024; 
uint8_t tensor_arena[kTensorArenaSize] __attribute__((aligned(16)));

tflite::MicroInterpreter* interpreter = nullptr;
tflite::ErrorReporter* error_reporter = nullptr; // The crash reporter
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10); // Wait for monitor
  Serial.println("\n--- BOOTING ROBOT BRAIN ---");
  
  // 1. Setup Error Reporter (CRITICAL FOR DEBUGGING)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  TF_LITE_REPORT_ERROR(error_reporter, "Error Reporter Initialized.");

  // 2. Setup Pins
  setupPins();
  
  // 3. Load AI Model
  Serial.println("1. Loading Model...");
  const tflite::Model* model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter, "Schema Mismatch! Model: %d, Lib: %d", 
                         model->version(), TFLITE_SCHEMA_VERSION);
    // We continue anyway, sometimes it still works
  }

  // 4. Setup Interpreter
  Serial.println("2. Setting up Interpreter...");
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // 5. Allocate Memory
  Serial.println("3. Allocating Tensors...");
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed! Arena too small?");
    while(1); // Halt
  }

  // 6. Get Pointers
  Serial.println("4. Getting Input/Output Pointers...");
  input = interpreter->input(0);
  output = interpreter->output(0);
  
  Serial.println("✅ AI System Ready! 🧠");
  delay(1000);
}

void loop() {
  // --- PRIORITY 0: GOAL DETECTION (The New Logic) ---
  // Read IR Sensor. standard modules usually output LOW when Black is detected (check your specific module!)
  // You might need to adjust the potentiometer on the sensor module to get the sensitivity right.
  int goalDetected = digitalRead(irSensorPin); 
  
  if (goalDetected == HIGH) { // Change to LOW if your sensor logic is inverted
    Serial.println("🏆 GOAL REACHED! Stopping. 🏆");
    stopMotors();
    
    // Optional: Celebration Wiggle
    // motor(motorL1, motorL2, 1); delay(100);
    // motor(motorL1, motorL2, -1); delay(100);
    // stopMotors();
    
    while(1); // Freeze the robot here forever (until reset)
  }
  // --- STEP 1: READ SENSORS ---
  float dLeft = readDistance(trigPinL, echoPinL);
  float dFront = readDistance(trigPinF, echoPinF);
  float dRight = readDistance(trigPinR, echoPinR);

  // --- STEP 2: HYBRID SAFETY LAYER (The "Override") ---
  
  // Rule A: The "Cruise Control" (Fixes the spinning in open space)
  if (dLeft > 0.2 && dFront > 0.2 && dRight > 0.2) {
    Serial.println("Clear Path -> Forcing FORWARD");
    executeAction(0); // Force Forward
    delay(100);
    return; // Skip the AI, save battery
  }

  // Rule B: The "Emergency Brake" (Fixes the risky Left Wall behavior)
  if (dLeft < 0.08) { // Too close to Left Wall
    Serial.println("Left Wall Danger -> Reflex RIGHT");
    executeAction(2); // Turn Right
    delay(100);
    return;
  }
  if (dRight < 0.08) { // Too close to Right Wall
    Serial.println("Right Wall Danger -> Reflex LEFT");
    executeAction(1); // Turn Left
    delay(100);
    return;
  }
  if (dFront < 0.08) { // Too close to Front Wall
    Serial.println("Front Wall Danger -> Reflex TURN");
    executeAction(1); // Turn Left (Default escape)
    delay(100);
    return;
  }

  // --- STEP 3: AI NAVIGATION (For complex situations) ---
  // If we are in the "Middle Zone" (0.15m to 0.5m), let the AI decide.
  
  float inputL = constrain(dLeft / MAX_SENSOR_RANGE, 0.0, 1.0);
  float inputF = constrain(dFront / MAX_SENSOR_RANGE, 0.0, 1.0);
  float inputR = constrain(dRight / MAX_SENSOR_RANGE, 0.0, 1.0);

  // KEEP THE SWAP (It was correct!)
  input->data.f[0] = inputR; // Sim Index 0 was Right
  input->data.f[1] = inputF;
  input->data.f[2] = inputL; // Sim Index 2 was Left

  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed");
    return;
  }

  float q0 = output->data.f[0]; // Fwd
  float q1 = output->data.f[1]; // Left
  float q2 = output->data.f[2]; // Right

  int action = 0;
  if (q1 > q0 && q1 > q2) action = 1;
  if (q2 > q0 && q2 > q1) action = 2;

  Serial.print("AI Driving | L:"); Serial.print(inputL);
  Serial.print(" F:"); Serial.print(inputF); 
  Serial.print(" R:"); Serial.print(inputR);
  Serial.print(" -> Act:"); Serial.println(action);

  executeAction(action);
  delay(100);
}

// --- HELPERS ---
void setupPins() {
  pinMode(trigPinL, OUTPUT); pinMode(echoPinL, INPUT);
  pinMode(trigPinF, OUTPUT); pinMode(echoPinF, INPUT);
  pinMode(trigPinR, OUTPUT); pinMode(echoPinR, INPUT);
  pinMode(motorL1, OUTPUT); pinMode(motorL2, OUTPUT);
  pinMode(motorR1, OUTPUT); pinMode(motorR2, OUTPUT);
  pinMode(irSensorPin, INPUT);
  stopMotors();
}

float readDistance(int trig, int echo) {
  digitalWrite(trig, LOW); delayMicroseconds(2);
  digitalWrite(trig, HIGH); delayMicroseconds(10);
  digitalWrite(trig, LOW);
  long duration = pulseIn(echo, HIGH, 30000); 
  if (duration == 0) return MAX_SENSOR_RANGE;
  float d = (duration * 0.034 / 2) / 100.0;
  return (d > MAX_SENSOR_RANGE) ? MAX_SENSOR_RANGE : d;
}

void executeAction(int action) {
  if (action == 0) { motor(motorL1, motorL2, 1); motor(motorR1, motorR2, 1); } // Fwd
  else if (action == 1) { motor(motorL1, motorL2, -1); motor(motorR1, motorR2, 1); } // Left
  else if (action == 2) { motor(motorL1, motorL2, 1); motor(motorR1, motorR2, -1); } // Right
}

void motor(int pin1, int pin2, int dir) {
  if (dir == 1) { digitalWrite(pin1, HIGH); digitalWrite(pin2, LOW); }
  else if (dir == -1) { digitalWrite(pin1, LOW); digitalWrite(pin2, HIGH); }
  else { digitalWrite(pin1, LOW); digitalWrite(pin2, LOW); }
}

void stopMotors() { motor(motorL1, motorL2, 0); motor(motorR1, motorR2, 0); }