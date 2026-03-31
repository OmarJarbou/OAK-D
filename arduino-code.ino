#include <SPI.h>
#include <MFRC522.h>
#include <Servo.h>
#include <Wire.h>
#include "Adafruit_TCS34725.h"
#include <math.h>

// ================= RFID =================
#define SS_PIN 9
#define RST_PIN 8
MFRC522 mfrc522(SS_PIN, RST_PIN);

byte allowedUID[4] = {0xD3, 0x60, 0xEA, 0x1A};

// ================= Brake + Vibration =================
Servo RightArmServo;
Servo LeftArmServo;

const int motorPin = 7;   // vibration motor via MOS module

// servo positions
const int RIGHT_RELEASE = 180;
const int LEFT_RELEASE  = 30;
const int RIGHT_BRAKE   = 140;
const int LEFT_BRAKE    = 70;

// ================= RGB + IR =================
const int IR_PIN = 10;
const int LED_PIN = 6;

bool objectDetected = false;

Adafruit_TCS34725 tcs =
Adafruit_TCS34725(TCS34725_INTEGRATIONTIME_154MS, TCS34725_GAIN_4X);

const int NUM_SAMPLES = 7;

// ================= System State =================
bool authorized = false;
bool tcsFound = false;

// ================= Startup Protection =================
unsigned long startupTime = 0;
const unsigned long STARTUP_IGNORE_TIME = 1500;

// ================= Non-blocking auth action sequence =================
// After valid RFID:
// 4 times:
//   motor ON + brake ON  for 2 sec
//   motor OFF + release  for 2 sec
bool authSequenceActive = false;
bool authSequenceOutputOn = false;
unsigned long authSequenceLastChange = 0;
int authSequenceCyclesDone = 0;

const unsigned long ACTION_ON_TIME  = 2000;
const unsigned long ACTION_OFF_TIME = 2000;
const int ACTION_TOTAL_CYCLES = 4;

// ================= Banknote Profiles =================
struct NoteProfile {
  const char* name;
  float r;
  float g;
  float b;
};

NoteProfile notes[] = {
  {"20 ILS",  0.415, 0.310, 0.301},
  {"50 ILS",  0.261, 0.418, 0.324},
  {"100 ILS", 0.328, 0.369, 0.311},
  {"200 ILS", 0.227, 0.396, 0.413}
};

const int NUM_NOTES = sizeof(notes) / sizeof(notes[0]);
const float MATCH_THRESHOLD = 0.085;

// ================= Functions =================
bool checkUID(byte *readUID, byte *validUID, byte size) {
  for (byte i = 0; i < size; i++) {
    if (readUID[i] != validUID[i]) {
      return false;
    }
  }
  return true;
}

void brakeRelease() {
  RightArmServo.write(RIGHT_RELEASE);
  LeftArmServo.write(LEFT_RELEASE);
}

void brakePush() {
  RightArmServo.write(RIGHT_BRAKE);
  LeftArmServo.write(LEFT_BRAKE);
}

void outputsOff() {
  brakeRelease();
  digitalWrite(motorPin, LOW);
}

void outputsOn() {
  brakePush();
  digitalWrite(motorPin, HIGH);
}

void startAuthSequence() {
  authSequenceActive = true;
  authSequenceOutputOn = true;
  authSequenceCyclesDone = 0;
  authSequenceLastChange = millis();

  outputsOn();
  Serial.println("Auth action sequence started");
}

void stopAuthSequence() {
  // do nothing if already stopped
  if (!authSequenceActive && !authSequenceOutputOn) {
    outputsOff();
    return;
  }

  authSequenceActive = false;
  authSequenceOutputOn = false;
  authSequenceCyclesDone = 0;
  outputsOff();
  Serial.println("Auth action sequence finished");
}

float colorDistanceWeighted(float r1, float g1, float b1, float r2, float g2, float b2) {
  float dr = r1 - r2;
  float dg = g1 - g2;
  float db = b1 - b2;

  return sqrt((1.3 * dr * dr) + (1.0 * dg * dg) + (1.3 * db * db));
}

bool readAverageColor(float &red, float &green, float &blue,
                      uint16_t &avgR, uint16_t &avgG, uint16_t &avgB, uint16_t &avgC) {
  unsigned long sumR = 0;
  unsigned long sumG = 0;
  unsigned long sumB = 0;
  unsigned long sumC = 0;
  int validSamples = 0;

  digitalWrite(LED_PIN, HIGH);
  delay(120);

  for (int i = 0; i < NUM_SAMPLES; i++) {
    uint16_t r, g, b, c;
    tcs.getRawData(&r, &g, &b, &c);

    if (c > 0) {
      sumR += r;
      sumG += g;
      sumB += b;
      sumC += c;
      validSamples++;
    }

    delay(40);
  }

  digitalWrite(LED_PIN, LOW);

  if (validSamples == 0) return false;

  avgR = sumR / validSamples;
  avgG = sumG / validSamples;
  avgB = sumB / validSamples;
  avgC = sumC / validSamples;

  if (avgC == 0) return false;

  red   = (float)avgR / avgC;
  green = (float)avgG / avgC;
  blue  = (float)avgB / avgC;

  return true;
}

const char* classifyBanknote(float red, float green, float blue, float &bestDistance) {
  int bestIndex = -1;
  float smallest = 999.0;
  float secondSmallest = 999.0;

  for (int i = 0; i < NUM_NOTES; i++) {
    float d = colorDistanceWeighted(red, green, blue, notes[i].r, notes[i].g, notes[i].b);

    if (d < smallest) {
      secondSmallest = smallest;
      smallest = d;
      bestIndex = i;
    } else if (d < secondSmallest) {
      secondSmallest = d;
    }
  }

  bestDistance = smallest;

  if (bestIndex == -1 || smallest > MATCH_THRESHOLD) {
    return "Unknown banknote";
  }

  if ((secondSmallest - smallest) < 0.015) {
    return "Ambiguous banknote";
  }

  return notes[bestIndex].name;
}

bool confirmObjectPresent() {
  int lowCount = 0;

  for (int i = 0; i < 5; i++) {
    if (digitalRead(IR_PIN) == LOW) {
      lowCount++;
    }
    delay(10);
  }

  return (lowCount >= 4);
}

bool confirmObjectRemoved() {
  int highCount = 0;

  for (int i = 0; i < 5; i++) {
    if (digitalRead(IR_PIN) == HIGH) {
      highCount++;
    }
    delay(10);
  }

  return (highCount >= 4);
}

void handleRFID() {
  if (mfrc522.PICC_IsNewCardPresent() && mfrc522.PICC_ReadCardSerial()) {

    Serial.print("UID: ");
    for (byte i = 0; i < mfrc522.uid.size; i++) {
      if (mfrc522.uid.uidByte[i] < 0x10) Serial.print("0");
      Serial.print(mfrc522.uid.uidByte[i], HEX);
      Serial.print(" ");
    }
    Serial.println();

    if (checkUID(mfrc522.uid.uidByte, allowedUID, 4)) {
      authorized = true;
      objectDetected = false;

      Serial.println("Valid card - system enabled");

      // start the 4-times motor/brake sequence
      startAuthSequence();
    } else {
      authorized = false;
      objectDetected = false;

      Serial.println("Invalid card - system disabled");

      stopAuthSequence();
    }

    mfrc522.PICC_HaltA();
    mfrc522.PCD_StopCrypto1();
    delay(150);
  }
}

void handleAuthSequence() {
  if (!authorized) {
    if (authSequenceActive || authSequenceOutputOn) {
      stopAuthSequence();
    } else {
      outputsOff();
    }
    return;
  }

  if (!authSequenceActive) {
    return;
  }

  unsigned long now = millis();

  // ON phase
  if (authSequenceOutputOn) {
    if (now - authSequenceLastChange >= ACTION_ON_TIME) {
      outputsOff();
      authSequenceOutputOn = false;
      authSequenceLastChange = now;
      authSequenceCyclesDone++;

      Serial.print("Cycle completed: ");
      Serial.println(authSequenceCyclesDone);

      if (authSequenceCyclesDone >= ACTION_TOTAL_CYCLES) {
        stopAuthSequence();
      }
    }
  }
  // OFF phase
  else {
    if (authSequenceCyclesDone < ACTION_TOTAL_CYCLES &&
        now - authSequenceLastChange >= ACTION_OFF_TIME) {
      outputsOn();
      authSequenceOutputOn = true;
      authSequenceLastChange = now;
    }
  }
}

void handleBanknoteDetection() {
  if (!authorized) {
    objectDetected = false;
    return;
  }

  if (!tcsFound) {
    return;
  }

  if (millis() - startupTime < STARTUP_IGNORE_TIME) {
    return;
  }

  int irValue = digitalRead(IR_PIN);

  // detect new object only if confirmed LOW
  if (!objectDetected && irValue == LOW) {
    if (confirmObjectPresent()) {
      objectDetected = true;

      Serial.println("Object detected...");

      float red, green, blue;
      uint16_t avgR, avgG, avgB, avgC;

      bool ok = readAverageColor(red, green, blue, avgR, avgG, avgB, avgC);

      if (!ok) {
        Serial.println("Failed to read valid color data");
        Serial.println("---------------------");
      } else {
        Serial.print("AVG R: "); Serial.print(avgR);
        Serial.print(" G: "); Serial.print(avgG);
        Serial.print(" B: "); Serial.print(avgB);
        Serial.print(" C: "); Serial.println(avgC);

        Serial.print("Normalized R: "); Serial.print(red, 4);
        Serial.print(" G: "); Serial.print(green, 4);
        Serial.print(" B: "); Serial.println(blue, 4);

        Serial.print("R-G diff: "); Serial.println(red - green, 4);
        Serial.print("B-G diff: "); Serial.println(blue - green, 4);

        float bestDistance;
        const char* result = classifyBanknote(red, green, blue, bestDistance);

        Serial.print("Detected: ");
        Serial.println(result);

        Serial.print("Match distance: ");
        Serial.println(bestDistance, 4);

        Serial.println("---------------------");
      }
    }
  }

  // reset when object is really removed
  if (objectDetected && irValue == HIGH) {
    if (confirmObjectRemoved()) {
      objectDetected = false;
      Serial.println("Object removed");
      Serial.println("---------------------");
    }
  }
}

void setup() {
  Serial.begin(9600);
  delay(200);

  startupTime = millis();

  // Important for Mega SPI
  pinMode(53, OUTPUT);
  digitalWrite(53, HIGH);

  // motor
  pinMode(motorPin, OUTPUT);
  digitalWrite(motorPin, LOW);

  // IR + sensor LED pin
  pinMode(IR_PIN, INPUT);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // servos
  RightArmServo.attach(22);
  LeftArmServo.attach(23);
  brakeRelease();

  // RFID
  SPI.begin();
  mfrc522.PCD_Init();

  delay(100);
  Serial.println("RC522 test start...");
  Serial.print("Reader Version: 0x");
  byte v = mfrc522.PCD_ReadRegister(mfrc522.VersionReg);
  Serial.println(v, HEX);

  if (v == 0x00 || v == 0xFF) {
    Serial.println("RFID module not connected or wiring is wrong");
  } else {
    Serial.println("RFID module detected. Show your card...");
  }

  // TCS34725
  if (tcs.begin()) {
    tcsFound = true;
    Serial.println("TCS34725 sensor found");
    tcs.setInterrupt(true);
  } else {
    tcsFound = false;
    Serial.println("TCS34725 sensor NOT found");
  }

  authorized = false;
  authSequenceActive = false;
  authSequenceOutputOn = false;
  authSequenceCyclesDone = 0;
  outputsOff();

  Serial.println("System locked until valid RFID tag");
  Serial.println("System ready...");
  Serial.println("---------------------");
}

void loop() {
  handleRFID();             // always listen for card
  handleAuthSequence();     // non-blocking motor + servo sequence
  handleBanknoteDetection();// works after authorization, even during sequence
}

// #include <SPI.h>
// #include <MFRC522.h>

// #define SS_PIN 9
// #define RST_PIN 8

// MFRC522 mfrc522(SS_PIN, RST_PIN);

// void setup() {
//   Serial.begin(9600);

//   pinMode(53, OUTPUT);
//   digitalWrite(53, HIGH);

//   SPI.begin();
//   mfrc522.PCD_Init();
//   delay(100);

//   Serial.print("Reader Version: 0x");
//   byte v = mfrc522.PCD_ReadRegister(mfrc522.VersionReg);
//   Serial.println(v, HEX);

//   if (v == 0x00 || v == 0xFF) {
//     Serial.println("الموديول غير متصل أو التوصيل غلط");
//   } else {
//     Serial.println("الموديول ظاهر");
//   }
// }

// void loop() {
// }

// const int rfPin = 11;  // receiver DATA pin
// const int buzzerPin = 12; // buzzer connected to pin 12

// void setup() {
//   Serial.begin(9600);
//   pinMode(rfPin, INPUT);
//   pinMode(buzzerPin, OUTPUT);  // set buzzer pin as output
//   Serial.println("RF Test Started...");
// }

// void loop() {
//   int state = digitalRead(rfPin);
//   Serial.println(state);  // prints 0 or 1
//   if (state == 1) {
//     digitalWrite(buzzerPin, HIGH); // buzzer ON
//     delay(1000);
//   }
//   digitalWrite(buzzerPin, LOW);  // buzzer OFF
//   delay(1000);  
// }

// #include <Wire.h>
// #include "Adafruit_TCS34725.h"
// #include <math.h>

// const int IR_PIN = 10;
// const int LED_PIN = 6;

// bool objectDetected = false;

// Adafruit_TCS34725 tcs =
// Adafruit_TCS34725(TCS34725_INTEGRATIONTIME_154MS, TCS34725_GAIN_4X);

// const int NUM_SAMPLES = 7;

// // Updated from your real readings
// struct NoteProfile {
//   const char* name;
//   float r;
//   float g;
//   float b;
// };

// NoteProfile notes[] = {
//   {"20 ILS",  0.415, 0.310, 0.301},  // strongest valid 20 sample
//   {"50 ILS",  0.261, 0.418, 0.324},
//   {"100 ILS", 0.328, 0.369, 0.311},
//   {"200 ILS", 0.227, 0.396, 0.413}
// };

// const int NUM_NOTES = sizeof(notes) / sizeof(notes[0]);

// // Bigger = more tolerant, smaller = stricter
// const float MATCH_THRESHOLD = 0.085;

// // ---------- Weighted color distance ----------
// float colorDistanceWeighted(float r1, float g1, float b1, float r2, float g2, float b2) {
//   float dr = r1 - r2;
//   float dg = g1 - g2;
//   float db = b1 - b2;

//   // blue is useful for 200, red useful for 20
//   return sqrt((1.3 * dr * dr) + (1.0 * dg * dg) + (1.3 * db * db));
// }

// // ---------- Read averaged RGB ----------
// bool readAverageColor(float &red, float &green, float &blue,
//                       uint16_t &avgR, uint16_t &avgG, uint16_t &avgB, uint16_t &avgC) {
//   unsigned long sumR = 0;
//   unsigned long sumG = 0;
//   unsigned long sumB = 0;
//   unsigned long sumC = 0;
//   int validSamples = 0;

//   digitalWrite(LED_PIN, HIGH);
//   delay(150);

//   for (int i = 0; i < NUM_SAMPLES; i++) {
//     uint16_t r, g, b, c;
//     tcs.getRawData(&r, &g, &b, &c);

//     if (c > 0) {
//       sumR += r;
//       sumG += g;
//       sumB += b;
//       sumC += c;
//       validSamples++;
//     }

//     delay(60);
//   }

//   digitalWrite(LED_PIN, LOW);

//   if (validSamples == 0) return false;

//   avgR = sumR / validSamples;
//   avgG = sumG / validSamples;
//   avgB = sumB / validSamples;
//   avgC = sumC / validSamples;

//   if (avgC == 0) return false;

//   red   = (float)avgR / avgC;
//   green = (float)avgG / avgC;
//   blue  = (float)avgB / avgC;

//   return true;
// }

// // ---------- Classify banknote ----------
// const char* classifyBanknote(float red, float green, float blue, float &bestDistance) {
//   int bestIndex = -1;
//   int secondIndex = -1;

//   float smallest = 999.0;
//   float secondSmallest = 999.0;

//   for (int i = 0; i < NUM_NOTES; i++) {
//     float d = colorDistanceWeighted(red, green, blue, notes[i].r, notes[i].g, notes[i].b);

//     if (d < smallest) {
//       secondSmallest = smallest;
//       secondIndex = bestIndex;

//       smallest = d;
//       bestIndex = i;
//     } else if (d < secondSmallest) {
//       secondSmallest = d;
//       secondIndex = i;
//     }
//   }

//   bestDistance = smallest;

//   if (bestIndex == -1 || smallest > MATCH_THRESHOLD) {
//     return "Unknown banknote";
//   }

//   // If two notes are too close, don't force a bad answer
//   if ((secondSmallest - smallest) < 0.015) {
//     return "Ambiguous banknote";
//   }

//   return notes[bestIndex].name;
// }

// void setup() {
//   Serial.begin(9600);

//   pinMode(IR_PIN, INPUT);
//   pinMode(LED_PIN, OUTPUT);
//   digitalWrite(LED_PIN, LOW);

//   if (tcs.begin()) {
//     Serial.println("TCS34725 sensor found");
//     tcs.setInterrupt(true);
//   } else {
//     Serial.println("Sensor not found");
//     while (1);
//   }
// }

// void loop() {
//   int irValue = digitalRead(IR_PIN);

//   if (irValue == LOW && !objectDetected) {
//     objectDetected = true;

//     Serial.println("Object detected...");
//     delay(250);

//     float red, green, blue;
//     uint16_t avgR, avgG, avgB, avgC;

//     bool ok = readAverageColor(red, green, blue, avgR, avgG, avgB, avgC);

//     if (!ok) {
//       Serial.println("Failed to read valid color data");
//       Serial.println("---------------------");
//       return;
//     }

//     Serial.print("AVG R: "); Serial.print(avgR);
//     Serial.print(" G: "); Serial.print(avgG);
//     Serial.print(" B: "); Serial.print(avgB);
//     Serial.print(" C: "); Serial.println(avgC);

//     Serial.print("Normalized R: "); Serial.print(red, 4);
//     Serial.print(" G: "); Serial.print(green, 4);
//     Serial.print(" B: "); Serial.println(blue, 4);

//     // Helpful extra indicators
//     Serial.print("R-G diff: "); Serial.println(red - green, 4);
//     Serial.print("B-G diff: "); Serial.println(blue - green, 4);

//     float bestDistance;
//     const char* result = classifyBanknote(red, green, blue, bestDistance);

//     Serial.print("Detected: ");
//     Serial.println(result);

//     Serial.print("Match distance: ");
//     Serial.println(bestDistance, 4);

//     Serial.println("---------------------");
//   }

//   if (irValue == HIGH) {
//     objectDetected = false;
//   }

//   delay(50);
// }

// #include <Servo.h>

// Servo RightArmServo;
// Servo LeftArmServo;

// const int motorPin = 13;   // vibration motor (MOS SIG)

// long duration;
// int distance;

// void setup() {
//   Serial.begin(9600);

//   RightArmServo.attach(22);
//   LeftArmServo.attach(23);

//   pinMode(motorPin, OUTPUT);

//   // --- Test vibration motor at startup ---
//   digitalWrite(motorPin, HIGH);   // motor ON
//   delay(2000);                    // run for 2 seconds
//   digitalWrite(motorPin, LOW);    // motor OFF
// }

// void loop() {
//   RightArmServo.write(140);   // push
//   LeftArmServo.write(70);   // push

//   delay(2000);

//   RightArmServo.write(180);   // release
//   LeftArmServo.write(30);   // release

//   delay(1000);
// }
