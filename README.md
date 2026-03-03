```mermaid
    graph TD
    %% Main System State
    Start((System Start)) --> Wait[Listen for Python Serial Command]
    Wait --> Mode{Command Received?}

    %% Training Branch
    Mode -->|'T'| Train[Training Mode]
    Train --> T_Timer{10ms Passed? <br> 100Hz}
    T_Timer -->|Yes| T_Read[Read EMGs A0-A3]
    T_Read --> T_Send[Send Raw EMG String]
    T_Send --> T_Timer

    %% Inference Branch
    Mode -->|'I'| Infer[Inference Mode]
    Infer --> I_Servo{Servo Command?}
    I_Servo -->|'O'| Open[Open Hand]
    I_Servo -->|'C'| Close[Close Hand]

    Infer --> I_Timer{50ms Passed? <br> 20Hz}
    
    %% FSR Logic
    I_Timer -->|Yes| FSR_Read[Read FSRs A4-A5]
    FSR_Read --> FSR_Thresh1{Force > 50?}
    
    FSR_Thresh1 -->|No| FSR_Thresh2{Force < 30?}
    FSR_Thresh2 -->|Yes| Reset[Reset State & Timer]
    
    FSR_Thresh1 -->|Yes| RateCalc[Calculate Derivative: rate = dF/dt]
    RateCalc --> TimeCheck{Grip Time > 300ms?}
    
    TimeCheck -->|Yes| PeakCheck{Peak Rate > 2.0?}
    PeakCheck -->|Yes| Hard[Classify: HARD]
    PeakCheck -->|No| Soft[Classify: SOFT]

    %% Haptic Logic
    RateCalc -.-> Haptic[Haptic Motor Logic]
    Haptic --> Haptic_Thresh{Max Force > 20?}
    Haptic_Thresh -->|Yes| Map[Map Delay: 60ms to 500ms]
    Map --> Pulse[Pulse FSR Motor: 80ms ON]

    %% Thermistor Logic
    RateCalc -.-> Therm_Read[Read Thermistors A6-A7]
    Therm_Read --> Therm_Thresh{Temp > 30°C <br> OR <br> Temp < 5°C?}
    Therm_Thresh -->|Yes| Therm_Pulse[Thermal Motor ON]

    %% Output
    Hard -.-> Telemetry[Send Telemetry String to Python]
    Soft -.-> Telemetry
    Pulse -.-> Telemetry
    Therm_Pulse -.-> Telemetry
    Telemetry --> I_Timer

    %% Styling for poster contrast
    classDef train fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef infer fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef logic fill:#fff3e0,stroke:#e65100,stroke-width:1px;
    
    class Train,T_Timer,T_Read,T_Send train;
    class Infer,I_Servo,I_Timer,Open,Close infer;
    class FSR_Read,FSR_Thresh1,FSR_Thresh2,RateCalc,TimeCheck,PeakCheck,Hard,Soft logic;
