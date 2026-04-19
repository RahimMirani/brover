# Brover — Project Plan

Voice-controlled AI agent embedded in an RC car. Speak a command into a phone web app; a Raspberry Pi onboard the car interprets it through the Claude API, reasons over camera input, and drives the motors to carry it out.

---

## 1. Project Overview

Brover is a physical AI agent project. An RC car has its stock control electronics replaced with a Raspberry Pi, a camera, and a motor driver. A web app served by the Pi provides the user-facing interface (voice input, live camera view, status). User voice commands are transcribed, sent to the Pi, and forwarded to the LLM API with a set of motor-control and sensing tools defined. Claude decides which tools to call and in what sequence. The Pi executes the tool calls on the physical hardware and streams results back to the user.

---

## 2. Goals

**Primary (MVP):**
- User can open a web page on their phone over the local network, speak a command ("go forward for two seconds, then turn left, or go to the kitchen and see if there is anyone there"), and see the car carry it out.
- Web page shows a live camera feed from the car.
- Claude receives the user's text command plus (optionally) a camera frame, and returns a sequence of tool calls.
- The Pi executes those tool calls on the motors.
- We will be using a llm with an api key rather than running the local llm on the pi. 

---

## 3. Hardware

| Component | Notes |
|---|---|
| Raspberry Pi | Pi 5 4gb |
| Camera | Connected via CSI ribbon cable, camera is IMX708|
| RC car chassis | Amazon B0DJ7BT1V5. Stock radio receiver is bypassed — motors wired directly to the motor driver. |
| Motor driver | L298N, DRV8833, or TB6612FNG. GPIO-controlled from the Pi. |
| Power | **Pi and motors must be on separate power rails with a common ground.** Motor current spikes will brown out the Pi otherwise. Typical setup: USB power bank for the Pi, 2 18650 li-ion batteries rechargeable pack for motors. |
| Phone | Any phone

---

## 4. System Architecture

Three logical components communicating over the network:

```
┌──────────────┐      WebSocket       ┌──────────────┐      HTTPS        ┌──────────────┐
│    Phone     │ ───────────────────▶ │ Raspberry Pi │ ────────────────▶ │  LLM 
│  (Web App)   │ ◀─────────────────── │  (FastAPI)   │ ◀──────────────── │  API      │
└──────────────┘   commands + audio   └──────┬───────┘   tool calls      └──────────────┘
                                             │
                                             ▼
                                     ┌──────────────┐
                                     │   Hardware   │
                                     │ (motors,     │
                                     │  camera)     │
                                     └──────────────┘
```

**Flow per command:**
1. User speaks into phone → browser's or other Web Speech API transcribes to text, what do you think?
2. Phone sends text over WebSocket to Pi.
3. Pi grabs a camera frame (if relevant), calls llm API with text + image + tool definitions.
4. LLM returns one or more tool calls (e.g. `forward(seconds=2)`, `turn(degrees=-30)`).
5. Pi executes each tool call, sends status back to phone over WebSocket.
6. Loop continues if Claude requests more actions or more sensor input.


## GPIO pins connected to L298N
IN1 = OutputDevice(17)   # Left motor input 1
IN2 = OutputDevice(27)   # Left motor input 2
IN3 = OutputDevice(22)   # Right motor input 1
IN4 = OutputDevice(23)   # Right motor input 2
