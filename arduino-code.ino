// ================================================================
//  Smart Walker — Full Integration v1.0
//  Arduino Mega
//
//  Serial  (USB, 9600) = debug monitor
//  Serial1 (pins 18/19)  = Pi / camera communication
//
//  Subsystems:
//    - RFID authorization gate
//    - Brake servos + vibration motor
//    - Stepper + potentiometer steering (7 positions)
//    - Banknote detection (TCS34725 + IR)
//    - FREE / ASSIST mode
// ================================================================

// ┌──────────────────────────────────────────────────┐
// │              Smart Walker v1.0                   │
// ├──────────────┬───────────────────────────────────┤
// │ Serial       │ USB — debug, calibration, status  │
// │ Serial1      │ Pi — CMD:xx in / STATUS:xx out    │
// ├──────────────┼───────────────────────────────────┤
// │ RFID gate    │ locks everything until authorized │
// │ Auth seq     │ 4× brake+vibrate on card scan     │
// │ Steering     │ pot closed-loop, 7 positions      │
// │ Brake        │ CMD:STOP / CMD:BRAKE:ON/OFF       │
// │ Banknote     │ runs silently, sends BANK:result  │
// └──────────────┴───────────────────────────────────┘

// 🔹 Pi → Arduino Commands
// | Pi Sends        | Arduino Does                            |
// | --------------- | --------------------------------------- |
// | `CMD:STOP`      | ASSIST mode + brake + vibration         |
// | `CMD:FREE`      | Release motor (user walks freely)       |
// | `CMD:ASSIST`    | Engage motor (steering active)          |
// | `CMD:GO:LEFT`   | Steer to full left                      |
// | `CMD:GO:L2`     | Steer to 2/3 left                       |
// | `CMD:GO:L1`     | Steer to 1/3 left                       |
// | `CMD:GO:CENTER` | Steer to center                         |
// | `CMD:GO:R1`     | Steer to 1/3 right                      |
// | `CMD:GO:R2`     | Steer to 2/3 right                      |
// | `CMD:GO:RIGHT`  | Steer to full right                     |
// | `CMD:BRAKE:ON`  | Activate brake only (no vibration stop) |
// | `CMD:BRAKE:OFF` | Release brake                           |
// | `CMD:UNLOCK`    | Unlock stepper limits                   |

// 🔹 Arduino → Pi Status Messages
// | Message               | Meaning                              |
// | --------------------- | ------------------------------------ |
// | `STATUS:AUTHORIZED`   | Valid RFID card scanned              |
// | `STATUS:UNAUTHORIZED` | Invalid RFID card                    |
// | `STATUS:MOVING:LEFT`  | Steering in progress                 |
// | `STATUS:REACHED`      | Target position reached              |
// | `STATUS:STOPPED`      | Emergency stop executed              |
// | `STATUS:FREE`         | Switched to FREE mode                |
// | `STATUS:ASSIST`       | Switched to ASSIST mode              |
// | `STATUS:SENSOR_ERROR` | Potentiometer disconnected / invalid |
// | `BANK:20 ILS`         | Banknote detected and identified     |
// | `BANK:REMOVED`        | Banknote removed                     |


#include <SPI.h>
#include <MFRC522.h>
#include <Servo.h>
#include <Wire.h>
#include "Adafruit_TCS34725.h"
#include <EEPROM.h>
#include <math.h>

// ================================================================
//  PIN DEFINITIONS
// ================================================================

// RFID
#define SS_PIN   9
#define RST_PIN  8

// Stepper
#define STEP_PIN 30
#define DIR_PIN  31
#define EN_PIN   32
#define POT_PIN  A0

// Brake + Vibration
#define MOTOR_PIN     7
#define RIGHT_SERVO_PIN 22
#define LEFT_SERVO_PIN  23

// Banknote
#define IR_PIN   10
#define LED_PIN  6

// ================================================================
//  EEPROM
// ================================================================
#define EEPROM_VALID_ADDR  0
#define EEPROM_CENTER_ADDR 2
#define EEPROM_LEFT_ADDR   4
#define EEPROM_RIGHT_ADDR  6
#define EEPROM_MAGIC       0xAB

// ================================================================
//  STEPPER TUNING
// ================================================================
const bool  INVERT_DIR      = false;
const int   POT_DEADBAND    = 6;
const int   MAX_STEPS       = 700;
const int   JOG_ADC         = 15;
const int   STEP_SLOW       = 1200;
const int   STEP_FAST       = 400;
const int   SPEED_THRESHOLD = 30;
const int   RAMP_STEPS      = 80;
const int   POT_MIN_VALID   = 50;
const int   POT_MAX_VALID   = 1000;

// ================================================================
//  BRAKE TUNING
// ================================================================
const int RIGHT_RELEASE = 180;
const int LEFT_RELEASE  = 30;
const int RIGHT_BRAKE   = 140;
const int LEFT_BRAKE    = 70;

// ================================================================
//  BANKNOTE TUNING
// ================================================================
const int   NUM_SAMPLES     = 7;
const float MATCH_THRESHOLD = 0.085;

// ================================================================
//  AUTH SEQUENCE TUNING
// ================================================================
const unsigned long ACTION_ON_TIME    = 2000;
const unsigned long ACTION_OFF_TIME   = 2000;
const int           ACTION_TOTAL_CYCLES = 2;
const unsigned long STARTUP_IGNORE_TIME = 1500;

// ================================================================
//  OBJECTS
// ================================================================
MFRC522 mfrc522(SS_PIN, RST_PIN);
Servo   RightArmServo;
Servo   LeftArmServo;
Adafruit_TCS34725 tcs =
  Adafruit_TCS34725(TCS34725_INTEGRATIONTIME_154MS, TCS34725_GAIN_4X);

// ================================================================
//  SYSTEM STATE
// ================================================================
bool authorized       = false;
bool tcsFound         = false;
unsigned long startupTime = 0;

// Auth sequence
bool  authSequenceActive    = false;
bool  authSequenceOutputOn  = false;
int   authSequenceCyclesDone = 0;
unsigned long authSequenceLastChange = 0;

// Banknote
bool objectDetected = false;

// Allowed RFID UID
byte allowedUID[4] = {0xD3, 0x60, 0xEA, 0x1A};

// ================================================================
//  STEPPER STATE
// ================================================================
int  CENTER_ADC = -1;
int  LEFT_ADC   = -1;
int  RIGHT_ADC  = -1;

bool lockedAtLeft  = false;
bool lockedAtRight = false;

enum Mode     { MODE_FREE, MODE_ASSIST };
Mode currentMode = MODE_FREE;
bool stopLatched = false;

// ================================================================
//  POSITION ENUM
// ================================================================
enum Position {
  POS_LEFT, POS_L2, POS_L1,
  POS_CENTER,
  POS_R1, POS_R2, POS_RIGHT,
  POS_UNKNOWN
};

const char* posNames[] = {
  "LEFT", "L2", "L1",
  "CENTER",
  "R1", "R2", "RIGHT"
};

// ================================================================
//  BANKNOTE PROFILES
// ================================================================
struct NoteProfile { const char* name; float r, g, b; };
NoteProfile notes[] = {
  {"20 ILS",  0.415, 0.310, 0.301},
  {"50 ILS",  0.261, 0.418, 0.324},
  {"100 ILS", 0.328, 0.369, 0.311},
  {"200 ILS", 0.227, 0.396, 0.413}
};
const int NUM_NOTES = sizeof(notes) / sizeof(notes[0]);

// ================================================================
//  Pi SERIAL BUFFER
// ================================================================
String piBuffer   = "";
String localBuffer = "";

// Forward declarations used before function definitions
void setFreeMode(bool sendStatus = true);
void setAssistMode(bool sendStatus = true);

// ================================================================
//  BRAKE / VIBRATION
// ================================================================
void brakeRelease() {
  RightArmServo.write(RIGHT_RELEASE);
  LeftArmServo.write(LEFT_RELEASE);
}

void brakePush() {
  RightArmServo.write(RIGHT_BRAKE);
  LeftArmServo.write(LEFT_BRAKE);
}

void vibrationPulse(unsigned long durationMs = 1000) {
  digitalWrite(MOTOR_PIN, HIGH); // vibration ON
  delay(durationMs);             // durationMs ms pulse
  digitalWrite(MOTOR_PIN, LOW);  // vibration OFF - brake stays ON
}

void outputsOn() {
  brakePush();                      // brake arms engage
  vibrationPulse(1000);
}

// ================================================================
//  AUTH SEQUENCE
// ================================================================
void startAuthSequence() {
  authSequenceActive     = true;
  authSequenceOutputOn   = true;
  authSequenceCyclesDone = 0;
  authSequenceLastChange = millis();
  outputsOn();
  Serial.println("[AUTH] Sequence started");
}

void stopAuthSequence(bool sendFreeStatus = true) {
  authSequenceActive     = false;
  authSequenceOutputOn   = false;
  authSequenceCyclesDone = 0;
  setFreeMode(false);   // enforce FREE state after auth sequence
  Serial.println("[AUTH] Sequence finished");
  if (sendFreeStatus) Serial1.println("STATUS:FREE");
}

void handleAuthSequence() {
  if (!authorized) {
    if (authSequenceActive || authSequenceOutputOn) stopAuthSequence(false);
    else brakeRelease();
    return;
  }

  if (!authSequenceActive) return;

  unsigned long now = millis();

  if (authSequenceOutputOn) {
    if (now - authSequenceLastChange >= ACTION_ON_TIME) {
      brakeRelease();
      authSequenceOutputOn   = false;
      authSequenceLastChange = now;
      authSequenceCyclesDone++;
      Serial.print("[AUTH] Cycle done: ");
      Serial.println(authSequenceCyclesDone);
      if (authSequenceCyclesDone >= ACTION_TOTAL_CYCLES) stopAuthSequence();
    }
  } else {
    if (authSequenceCyclesDone < ACTION_TOTAL_CYCLES &&
        now - authSequenceLastChange >= ACTION_OFF_TIME) {
      outputsOn();
      authSequenceOutputOn   = true;
      authSequenceLastChange = now;
    }
  }
}

// ================================================================
//  RFID
// ================================================================
bool checkUID(byte* readUID, byte* validUID, byte size) {
  for (byte i = 0; i < size; i++)
    if (readUID[i] != validUID[i]) return false;
  return true;
}

void handleRFID() {
  if (!mfrc522.PICC_IsNewCardPresent() || !mfrc522.PICC_ReadCardSerial()) return;

  Serial.print("[RFID] UID: ");
  for (byte i = 0; i < mfrc522.uid.size; i++) {
    if (mfrc522.uid.uidByte[i] < 0x10) Serial.print("0");
    Serial.print(mfrc522.uid.uidByte[i], HEX);
    Serial.print(" ");
  }
  Serial.println();

  if (checkUID(mfrc522.uid.uidByte, allowedUID, 4)) {
    authorized      = true;
    objectDetected  = false;
    Serial.println("[RFID] Authorized - system unlocked");
    Serial1.println("STATUS:AUTHORIZED");
    startAuthSequence();
  } else {
    authorized      = false;
    objectDetected  = false;
    Serial.println("[RFID] Denied - system locked");
    Serial1.println("STATUS:UNAUTHORIZED");
    if (authSequenceActive || authSequenceOutputOn) {
      stopAuthSequence(false);
    } else {
      setFreeMode(false);   // release wheel on deauth, no FREE protocol status
    }
  }

  mfrc522.PICC_HaltA();
  mfrc522.PCD_StopCrypto1();
  delay(150);
}

// ================================================================
//  STEPPER — SENSOR
// ================================================================
bool sensorValueValid(int val) {
  return val >= POT_MIN_VALID && val <= POT_MAX_VALID;
}

bool sensorOK(int val) {
  if (!sensorValueValid(val)) {
    Serial.print("[POT] SENSOR ERROR: ");
    Serial.println(val);
    Serial1.println("STATUS:SENSOR_ERROR");
    setFreeMode(false);
    return false;
  }
  return true;
}

int readPot() {
  long sum = 0;
  for (int i = 0; i < 8; i++) {
    sum += analogRead(POT_PIN);
    delayMicroseconds(200);
  }
  return (int)(sum / 8);
}

// ================================================================
//  STEPPER — MODE
// ================================================================
void setFreeMode(bool sendStatus) {
  currentMode   = MODE_FREE;
  stopLatched   = false;
  lockedAtLeft  = false;
  lockedAtRight = false;
  brakeRelease();
  digitalWrite(EN_PIN, HIGH);   // release motor
  Serial.println("[STEER] FREE MODE - wheel released");
  if (sendStatus) Serial1.println("STATUS:FREE");
}

void setAssistMode(bool sendStatus) {
  if (!authorized) {
    Serial.println("[STEER] Cannot enter ASSIST - not authorized");
    return;
  }
  currentMode = MODE_ASSIST;
  digitalWrite(EN_PIN, LOW);    // engage motor
  Serial.println("[STEER] ASSIST MODE - system steering");
  if (sendStatus) Serial1.println("STATUS:ASSIST");
}

// ================================================================
//  STEPPER — MOTION
// ================================================================
int speedForError(int error, int stepIndex) {
  int baseDelay;
  if (abs(error) > SPEED_THRESHOLD) {
    baseDelay = STEP_FAST;
  } else {
    baseDelay = map(abs(error), 0, SPEED_THRESHOLD, STEP_SLOW, STEP_FAST);
  }
  if (stepIndex < RAMP_STEPS) {
    int rd = STEP_SLOW - (STEP_SLOW - baseDelay) * stepIndex / RAMP_STEPS;
    return constrain(rd, STEP_FAST, STEP_SLOW);
  }
  return constrain(baseDelay, STEP_FAST, STEP_SLOW);
}

void lockMotor(bool atLeft) {
  if (atLeft) { lockedAtLeft  = true; Serial.println("[STEER] LOCKED LEFT"); }
  else        { lockedAtRight = true; Serial.println("[STEER] LOCKED RIGHT"); }
  Serial1.println(atLeft ? "STATUS:LOCKED_LEFT" : "STATUS:LOCKED_RIGHT");
  digitalWrite(EN_PIN, LOW);
}

void unlock() {
  lockedAtLeft  = false;
  lockedAtRight = false;
  Serial.println("[STEER] Unlocked");
  Serial1.println("STATUS:UNLOCKED");
}

bool isLocked(bool dirRight) {
  if (lockedAtLeft  && !dirRight) return true;
  if (lockedAtRight &&  dirRight) return true;
  return false;
}

void singleStep(bool dirRight, int delayUs) {
  bool realDir = INVERT_DIR ? !dirRight : dirRight;
  digitalWrite(DIR_PIN, realDir);
  digitalWrite(STEP_PIN, HIGH);
  delayMicroseconds(delayUs);
  digitalWrite(STEP_PIN, LOW);
  delayMicroseconds(delayUs);
}

bool moveToADC(int targetADC) {
  if (!authorized) {
    Serial.println("[STEER] Not authorized");
    return false;
  }
  if (currentMode == MODE_FREE) {
    Serial.println("[STEER] In FREE mode - send M:ASSIST first");
    return false;
  }

  int current = readPot();
  if (!sensorOK(current)) return false;

  int error = targetADC - current;
  if (abs(error) <= POT_DEADBAND) {
    Serial.println("[STEER] Already at target");
    Serial1.println("STATUS:AT_TARGET");
    return true;
  }

  bool dirRight = (error < 0);
  if (isLocked(dirRight)) return false;

  int safety = 0;
  while (safety < MAX_STEPS) {
    current = readPot();
    if (!sensorOK(current)) return false;

    error = targetADC - current;
    if (abs(error) <= POT_DEADBAND) {
      Serial.print("[STEER] Reached. potADC=");
      Serial.print(current);
      Serial.print(" target=");
      Serial.print(targetADC);
      Serial.print(" steps=");
      Serial.println(safety);
      Serial1.println("STATUS:REACHED");
      lockedAtLeft  = false;
      lockedAtRight = false;
      return true;
    }

    dirRight = (error < 0);
    if (isLocked(dirRight)) return false;

    singleStep(dirRight, speedForError(error, safety));
    safety++;
  }

  lockMotor(dirRight);
  return false;
}

// ================================================================
//  STEPPER — POSITIONS
// ================================================================
int adcForPosition(Position pos) {
  switch (pos) {
    case POS_LEFT:   return LEFT_ADC;
    case POS_L2:     return LEFT_ADC   + (CENTER_ADC - LEFT_ADC)  / 3;
    case POS_L1:     return LEFT_ADC   + (CENTER_ADC - LEFT_ADC)  * 2 / 3;
    case POS_CENTER: return CENTER_ADC;
    case POS_R1:     return CENTER_ADC + (RIGHT_ADC - CENTER_ADC) / 3;
    case POS_R2:     return CENTER_ADC + (RIGHT_ADC - CENTER_ADC) * 2 / 3;
    case POS_RIGHT:  return RIGHT_ADC;
    default:         return CENTER_ADC;
  }
}

int clampToRange(int targetADC) {
  if (LEFT_ADC == -1 || RIGHT_ADC == -1) return targetADC;
  return constrain(targetADC, min(LEFT_ADC, RIGHT_ADC), max(LEFT_ADC, RIGHT_ADC));
}

void goToPosition(Position pos) {
  if (CENTER_ADC == -1 || LEFT_ADC == -1 || RIGHT_ADC == -1) {
    Serial.println("[STEER] Not calibrated");
    Serial1.println("STATUS:NOT_CALIBRATED");
    return;
  }
  Serial.print("[STEER] Moving -> "); Serial.println(posNames[pos]);
  Serial1.print("STATUS:MOVING:"); Serial1.println(posNames[pos]);
  moveToADC(adcForPosition(pos));
}

void jogLeft() {
  int target = clampToRange(readPot() + JOG_ADC);
  Serial.print("[STEER] Jog LEFT -> "); Serial.println(target);
  moveToADC(target);
}

void jogRight() {
  int target = clampToRange(readPot() - JOG_ADC);
  Serial.print("[STEER] Jog RIGHT -> "); Serial.println(target);
  moveToADC(target);
}

// ================================================================
//  EEPROM
// ================================================================
void saveToEEPROM() {
  EEPROM.write(EEPROM_VALID_ADDR, EEPROM_MAGIC);
  EEPROM.put(EEPROM_CENTER_ADDR, CENTER_ADC);
  EEPROM.put(EEPROM_LEFT_ADDR,   LEFT_ADC);
  EEPROM.put(EEPROM_RIGHT_ADDR,  RIGHT_ADC);
  Serial.println("[EEPROM] Saved");
}

bool loadFromEEPROM() {
  if (EEPROM.read(EEPROM_VALID_ADDR) != EEPROM_MAGIC) return false;
  EEPROM.get(EEPROM_CENTER_ADDR, CENTER_ADC);
  EEPROM.get(EEPROM_LEFT_ADDR,   LEFT_ADC);
  EEPROM.get(EEPROM_RIGHT_ADDR,  RIGHT_ADC);
  return true;
}

// ================================================================
//  BANKNOTE DETECTION
// ================================================================
float colorDistanceWeighted(float r1, float g1, float b1,
                             float r2, float g2, float b2) {
  float dr = r1-r2, dg = g1-g2, db = b1-b2;
  return sqrt((1.3*dr*dr) + (1.0*dg*dg) + (1.3*db*db));
}

bool readAverageColor(float &red, float &green, float &blue,
                      uint16_t &avgR, uint16_t &avgG,
                      uint16_t &avgB, uint16_t &avgC) {
  unsigned long sumR=0, sumG=0, sumB=0, sumC=0;
  int validSamples = 0;

  digitalWrite(LED_PIN, HIGH);
  delay(120);

  for (int i = 0; i < NUM_SAMPLES; i++) {
    uint16_t r, g, b, c;
    tcs.getRawData(&r, &g, &b, &c);
    if (c > 0) { sumR+=r; sumG+=g; sumB+=b; sumC+=c; validSamples++; }
    delay(40);
  }
  digitalWrite(LED_PIN, LOW);

  if (validSamples == 0) return false;

  avgR = sumR/validSamples;
  avgG = sumG/validSamples;
  avgB = sumB/validSamples;
  avgC = sumC/validSamples;

  if (avgC == 0) return false;

  red   = (float)avgR / avgC;
  green = (float)avgG / avgC;
  blue  = (float)avgB / avgC;
  return true;
}

const char* classifyBanknote(float red, float green, float blue,
                              float &bestDistance) {
  int   bestIndex = -1;
  float smallest  = 999.0;
  float secondSmallest = 999.0;

  for (int i = 0; i < NUM_NOTES; i++) {
    float d = colorDistanceWeighted(red, green, blue,
                                    notes[i].r, notes[i].g, notes[i].b);
    if (d < smallest) {
      secondSmallest = smallest;
      smallest  = d;
      bestIndex = i;
    } else if (d < secondSmallest) {
      secondSmallest = d;
    }
  }

  bestDistance = smallest;
  if (bestIndex == -1 || smallest > MATCH_THRESHOLD) return "Unknown";
  if ((secondSmallest - smallest) < 0.015)           return "Ambiguous";
  return notes[bestIndex].name;
}

void handleBanknoteDetection() {
  if (!authorized || !tcsFound) return;
  if (millis() - startupTime < STARTUP_IGNORE_TIME) return;

  int irValue = digitalRead(IR_PIN);

  if (!objectDetected && irValue == LOW) {
    // Confirm object present
    int lowCount = 0;
    for (int i = 0; i < 5; i++) {
      if (digitalRead(IR_PIN) == LOW) lowCount++;
      delay(10);
    }
    if (lowCount < 4) return;

    objectDetected = true;
    Serial.println("[BANK] Object detected");

    float red, green, blue;
    uint16_t avgR, avgG, avgB, avgC;

    if (!readAverageColor(red, green, blue, avgR, avgG, avgB, avgC)) {
      Serial.println("[BANK] Color read failed");
      Serial1.println("BANK:ERROR");
      return;
    }

    float bestDistance;
    const char* result = classifyBanknote(red, green, blue, bestDistance);

    Serial.print("[BANK] Detected: "); Serial.println(result);
    Serial.print("[BANK] Distance: "); Serial.println(bestDistance, 4);

    // Send result to Pi
    Serial1.print("BANK:"); Serial1.println(result);
  }

  if (objectDetected && irValue == HIGH) {
    int highCount = 0;
    for (int i = 0; i < 5; i++) {
      if (digitalRead(IR_PIN) == HIGH) highCount++;
      delay(10);
    }
    if (highCount < 4) return;
    objectDetected = false;
    Serial.println("[BANK] Object removed");
    Serial1.println("BANK:REMOVED");
  }
}

// ================================================================
//  PRINT HELPERS (USB debug only)
// ================================================================
void printStatus() {
  int current = readPot();
  Serial.println("-------------------------------------");
  Serial.print("Authorized    = "); Serial.println(authorized  ? "YES" : "NO");
  Serial.print("Mode          = "); Serial.println(currentMode == MODE_FREE ? "FREE" : "ASSIST");
  Serial.print("Current potADC= "); Serial.println(current);
  Serial.print("Sensor valid  = "); Serial.println(sensorValueValid(current) ? "YES" : "NO");
  Serial.print("Auth sequence = "); Serial.println(authSequenceActive ? "RUNNING" : "IDLE");
  if (CENTER_ADC == -1) {
    Serial.println("NOT CALIBRATED");
  } else {
    for (int i = POS_LEFT; i <= POS_RIGHT; i++) {
      Serial.print("  "); Serial.print(posNames[i]);
      Serial.print("\t= ADC ");
      Serial.println(adcForPosition((Position)i));
    }
  }
  Serial.println("-------------------------------------");
}

void printHelp() {
  Serial.println("-- LOCAL COMMANDS (USB) --------------------");
  Serial.println("Mode   : M:FREE       M:ASSIST");
  Serial.println("Jog    : a=left       d=right");
  Serial.println("Go     : GO:LEFT/L2/L1/CENTER/R1/R2/RIGHT");
  Serial.println("Brake  : BRAKE:ON     BRAKE:OFF");
  Serial.println("Calib  : c=center     z=left    x=right");
  Serial.println("EEPROM : e=save");
  Serial.println("Unlock : u");
  Serial.println("Status : p            h=help");
  Serial.println("-- Pi COMMANDS (Serial1) -------------------");
  Serial.println("CMD:STOP");
  Serial.println("CMD:FREE");
  Serial.println("CMD:ASSIST");
  Serial.println("CMD:GO:LEFT/L2/L1/CENTER/R1/R2/RIGHT");
  Serial.println("CMD:BRAKE:ON / CMD:BRAKE:OFF");
  Serial.println("CMD:UNLOCK");
  Serial.println("--------------------------------------------");
}

// ================================================================
//  COMMAND EXECUTOR
//  Used by both USB and Pi — single source of truth
// ================================================================
void executeCommand(String cmd) {
  cmd.trim();
  cmd.toUpperCase();

  // ── Mode ──────────────────────────────────────────────────────
  if (cmd == "M:FREE"  || cmd == "CMD:FREE")   { setFreeMode();   return; }
  if (cmd == "M:ASSIST"|| cmd == "CMD:ASSIST") { setAssistMode(); return; }

  // ── Emergency stop ────────────────────────────────────────────
  if (cmd == "CMD:STOP") {
    setAssistMode(false);     // engage motor to hold, avoid extra STATUS:ASSIST
    brakePush();              // keep brake ON while stopped
    if (!stopLatched) {
      vibrationPulse(1000);   // pulse only on STOP transition
      stopLatched = true;
      Serial.println("[CMD] STOP - brake and vibration engaged");
      Serial1.println("STATUS:STOPPED");
    } else {
      Serial.println("[CMD] STOP already latched - skipping vibration");
    }
    return;
  }

  // ── Brake manual control ──────────────────────────────────────
  if (cmd == "BRAKE:ON"  || cmd == "CMD:BRAKE:ON")  {
    if (!authorized) { Serial.println("[CMD] Not authorized"); return; }
    brakePush();
    Serial.println("[CMD] Brake ON");
    return;
  }
  if (cmd == "BRAKE:OFF" || cmd == "CMD:BRAKE:OFF") {
    if (!authorized) { Serial.println("[CMD] Not authorized"); return; }
    stopLatched = false;
    brakeRelease();
    Serial.println("[CMD] Brake OFF");
    return;
  }

  // ── Unlock ────────────────────────────────────────────────────
  if (cmd == "U" || cmd == "CMD:UNLOCK") { unlock(); return; }

  // ── GO commands ───────────────────────────────────────────────
  // Accepts both GO:LEFT and CMD:GO:LEFT
  String goCmd = cmd;
  if (goCmd.startsWith("CMD:GO:")) {
    goCmd = goCmd.substring(4); // strip CMD:
  }

  if (goCmd.startsWith("GO:")) {
    if (!authorized) { Serial.println("[CMD] Not authorized"); return; }
    if (currentMode == MODE_FREE) {
      setAssistMode(true);   // enter ASSIST for steering and report mode change
    }
    stopLatched = false;
    brakeRelease();
    String pos = goCmd.substring(3);
    if      (pos == "LEFT")   goToPosition(POS_LEFT);
    else if (pos == "L2")     goToPosition(POS_L2);
    else if (pos == "L1")     goToPosition(POS_L1);
    else if (pos == "CENTER") goToPosition(POS_CENTER);
    else if (pos == "R1")     goToPosition(POS_R1);
    else if (pos == "R2")     goToPosition(POS_R2);
    else if (pos == "RIGHT")  goToPosition(POS_RIGHT);
    else { Serial.print("[CMD] Unknown position: "); Serial.println(pos); }
    return;
  }

  // ── Jog (USB debug only) ──────────────────────────────────────
  if (cmd.length() == 1) {
    switch (cmd[0]) {
      case 'A': jogLeft();  return;
      case 'D': jogRight(); return;
      case 'F': goToPosition(POS_CENTER); return;
      case 'C':
        CENTER_ADC = readPot();
        Serial.print("[CALIB] CENTER = "); Serial.println(CENTER_ADC);
        return;
      case 'Z':
        LEFT_ADC = readPot();
        Serial.print("[CALIB] LEFT = "); Serial.println(LEFT_ADC);
        return;
      case 'X':
        RIGHT_ADC = readPot();
        Serial.print("[CALIB] RIGHT = "); Serial.println(RIGHT_ADC);
        return;
      case 'E':
        if (CENTER_ADC==-1 || LEFT_ADC==-1 || RIGHT_ADC==-1)
          Serial.println("[EEPROM] Calibrate first");
        else saveToEEPROM();
        return;
      case 'P': printStatus(); return;
      case 'H': printHelp();   return;
    }
  }

  Serial.print("[CMD] Unknown: "); Serial.println(cmd);
}

// ================================================================
//  SERIAL HANDLERS
// ================================================================

// USB (Serial) — debug/calibration
void handleLocalSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (localBuffer.length() > 0) {
        executeCommand(localBuffer);
        localBuffer = "";
      }
    } else {
      localBuffer += c;
    }
  }
}

// Pi (Serial1) — camera commands
void handlePiSerial() {
  while (Serial1.available()) {
    char c = Serial1.read();
    if (c == '\n' || c == '\r') {
      if (piBuffer.length() > 0) {
    Serial.print("[Pi] Received: "); Serial.println(piBuffer);

        // Only execute if authorized and auth sequence done
        if (!authorized) {
          Serial.println("[Pi] Ignored - unauthorized");
          Serial1.println("STATUS:UNAUTHORIZED");
          piBuffer = "";
          return;
        }
        if (authSequenceActive) {
          Serial.println("[Pi] Ignored - not ready");
          Serial1.println("STATUS:NOT_READY");
          piBuffer = "";
          return;
        }

        executeCommand(piBuffer);
        piBuffer = "";
      }
    } else {
      piBuffer += c;
    }
  }
}

// ================================================================
//  SETUP
// ================================================================
void setup() {
  Serial.begin(9600);
  Serial1.begin(9600);
  delay(200);

  startupTime = millis();

  // Stepper
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN,  OUTPUT);
  pinMode(EN_PIN,   OUTPUT);
  pinMode(POT_PIN,  INPUT);

  // Brake + vibration
  pinMode(MOTOR_PIN, OUTPUT);
  digitalWrite(MOTOR_PIN, LOW);

  // Banknote
  pinMode(IR_PIN,  INPUT);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // Servos
  RightArmServo.attach(RIGHT_SERVO_PIN);
  LeftArmServo.attach(LEFT_SERVO_PIN);
  setFreeMode(false);   // safe default on boot (no protocol status before RFID)

  // RFID (Mega SPI fix)
  pinMode(53, OUTPUT);
  digitalWrite(53, HIGH);
  SPI.begin();
  mfrc522.PCD_Init();
  delay(100);

  byte v = mfrc522.PCD_ReadRegister(mfrc522.VersionReg);
  Serial.print("[RFID] Version: 0x"); Serial.println(v, HEX);
  if (v == 0x00 || v == 0xFF)
    Serial.println("[RFID] WARNING: Module not detected");
  else
    Serial.println("[RFID] Module OK");

  // TCS34725
  if (tcs.begin()) {
    tcsFound = true;
    tcs.setInterrupt(true);
    Serial.println("[TCS] Sensor found");
  } else {
    tcsFound = false;
    Serial.println("[TCS] Sensor NOT found");
  }

  // EEPROM
  if (loadFromEEPROM()) {
    Serial.println("[EEPROM] Calibration loaded");
    printStatus();
  } else {
    Serial.println("[EEPROM] No calibration - run calibration first");
  }

  authorized             = false;
  authSequenceActive     = false;
  authSequenceOutputOn   = false;
  authSequenceCyclesDone = 0;
  brakeRelease();

  Serial.println("=== Smart Walker v1.0 Ready ===");
  Serial.println("Waiting for RFID authorization...");
  printHelp();
}

// ================================================================
//  LOOP
// ================================================================
void loop() {
  handleLocalSerial();      // USB debug commands
  handlePiSerial();         // Pi/camera commands
  handleRFID();             // always listen for card
  handleAuthSequence();     // non-blocking brake/motor sequence
  handleBanknoteDetection();// runs after auth, independent
}