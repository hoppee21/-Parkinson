#include "mbed.h"
#include "math.h"
#include "arm_math.h"
#include "rtos/Thread.h"
#include <atomic>

// Serial over USB
BufferedSerial serial_port(USBTX, USBRX, 115200);
FileHandle *mbed::mbed_override_console(int) { return &serial_port; }

// I2C and sensor setup
I2C         i2c(PB_11, PB_10);
#define     LSM6DSL_ADDR   (0x6A << 1)
#define     WHO_AM_I       0x0F
#define     CTRL1_XL       0x10
#define     STATUS_REG     0x1E    // Accel & gyro status
#define     OUTX_L_XL      0x28
#define     OUTX_H_XL      0x29
#define     OUTY_L_XL      0x2A
#define     OUTY_H_XL      0x2B
#define     OUTZ_L_XL      0x2C
#define     OUTZ_H_XL      0x2D

// -----------------------------------------------------------------
// Globals for blinking
// -----------------------------------------------------------------
DigitalOut led1(LED1);
DigitalOut led2(LED2);
DigitalOut led3(LED3);

// Shared values written by FFT task, read by blink task
std::atomic<float> sharedPeakHz{0.0f};
std::atomic<float> sharedMaxVal{0.0f};

// Blink thread
rtos::Thread blinkThread;

// -----------------------------------------------------------------
// blink_task():
//   - Monitors `sharedPeakHz`
//   - Blinks LED1 when 3<ph≤5 Hz
//   - Blinks LED2 when 5<ph≤7 Hz
//   - Turns LEDs off immediately outside those bands
// -----------------------------------------------------------------
void blink_task() {
    using namespace std::chrono;

    Timer timer;
    timer.start();

    float last1 = 0.0f;  // timestamp of last toggle for LED1
    float last2 = 0.0f;  // timestamp of last toggle for LED2

    while (true) {
        // current time in seconds
        float now = duration<float>{ timer.elapsed_time() }.count();

        float ph = sharedPeakHz.load();

        if (ph > 3.0f && ph <= 5.0f) {
            // LED1 blinks at ph Hz
            float half_s = 0.5f / ph;  
            if (now - last1 >= half_s) {
                led1 = !led1;
                last1 = now;
            }
            // ensure LED2 is off (and reset its timer)
            led2 = 0;
            last2 = now;
        }
        else if (ph > 5.0f && ph <= 7.0f) {
            // LED2 blinks at ph Hz
            float half_s = 0.5f / ph;  
            if (now - last2 >= half_s) {
                led2 = !led2;
                last2 = now;
            }
            // ensure LED1 is off (and reset its timer)
            led1 = 0;
            last1 = now;
        }
        else {
            // outside both bands → both off immediately
            led1 = 0;
            led2 = 0;
            // reset both timers so we start fresh next time
            last1 = now;
            last2 = now;
        }

        // check again in 10 ms to react promptly
        ThisThread::sleep_for(10ms);
    }
}



// -----------------------------------------------------------------
// read_16bit_value(): Read and combine two 8‐bit registers into a
//   signed 16‐bit value.
// -----------------------------------------------------------------
void write_register(uint8_t reg, uint8_t value) {
    uint8_t data[2] = { reg, value };
    i2c.write(LSM6DSL_ADDR, (char*)data, 2);
}

uint8_t read_register(uint8_t reg) {
    uint8_t cmd = reg;
    i2c.write(LSM6DSL_ADDR, (char*)&cmd, 1, true);
    uint8_t resp;
    i2c.read(LSM6DSL_ADDR, (char*)&resp, 1);
    return resp;
}

int16_t read_16bit_value(uint8_t low_reg, uint8_t high_reg) {
    uint8_t lo = read_register(low_reg);
    uint8_t hi = read_register(high_reg);
    return (int16_t)((hi << 8) | lo);
}

// Button for start/stop
InterruptIn button(BUTTON1);
volatile bool isReading     = false;
volatile bool buttonPressed = false;
void button_pressed() { buttonPressed = true; }

// Sampling parameters
#define SAMPLE_RATE_HZ      416
#define RECORD_DURATION_S   3
#define RECORD_SIZE         (SAMPLE_RATE_HZ * RECORD_DURATION_S)  // 1248

// Buffers for recording
float32_t record_X[RECORD_SIZE];
float32_t record_Y[RECORD_SIZE];
float32_t record_Z[RECORD_SIZE];
uint16_t  record_count = 0;

// FFT parameters
#define FFT_SIZE            1024
float32_t sample_data_X[FFT_SIZE];
float32_t sample_data_Y[FFT_SIZE];
float32_t sample_data_Z[FFT_SIZE];

float32_t fft_out_X[FFT_SIZE];
float32_t fft_out_Y[FFT_SIZE];
float32_t fft_out_Z[FFT_SIZE];
float32_t magnitude_total[FFT_SIZE/2 + 1];
const float32_t SAMPLE_RATE = SAMPLE_RATE_HZ;

arm_rfft_fast_instance_f32 FFT_Instance;


// -----------------------------------------------------------------
// run_fft_all(): Compute the spectrum of the last FFT_SIZE samples.
//   - Demean
//   - Apply Hann window
//   - Perform RFFT
//   - Compute combined magnitude
// -----------------------------------------------------------------
void run_fft_all() {
    // De-mean
    float32_t meanX=0, meanY=0, meanZ=0;
    for(int j=0;j<FFT_SIZE;j++){
        meanX += sample_data_X[j];
        meanY += sample_data_Y[j];
        meanZ += sample_data_Z[j];
    }
    meanX /= FFT_SIZE; meanY /= FFT_SIZE; meanZ /= FFT_SIZE;
    for(int j=0;j<FFT_SIZE;j++){
        sample_data_X[j] -= meanX;
        sample_data_Y[j] -= meanY;
        sample_data_Z[j] -= meanZ;
    }
    // Hann window
    for(int j=0;j<FFT_SIZE;j++){
        float32_t w = 0.5f * (1.0f - cosf(2.0f * PI * j / (FFT_SIZE-1)));
        sample_data_X[j] *= w;
        sample_data_Y[j] *= w;
        sample_data_Z[j] *= w;
    }
    // RFFT
    arm_rfft_fast_f32(&FFT_Instance, sample_data_X, fft_out_X, 0);
    arm_rfft_fast_f32(&FFT_Instance, sample_data_Y, fft_out_Y, 0);
    arm_rfft_fast_f32(&FFT_Instance, sample_data_Z, fft_out_Z, 0);
    // magnitude
    for(int k=0;k<=FFT_SIZE/2;k++){
        float32_t magX, magY, magZ;
        if(k==0){
            magX = fabsf(fft_out_X[0]);
            magY = fabsf(fft_out_Y[0]);
            magZ = fabsf(fft_out_Z[0]);
        } else if(k==FFT_SIZE/2){
            magX = fabsf(fft_out_X[1]);
            magY = fabsf(fft_out_Y[1]);
            magZ = fabsf(fft_out_Z[1]);
        } else {
            float32_t rx=fft_out_X[2*k], ix=fft_out_X[2*k+1];
            float32_t ry=fft_out_Y[2*k], iy=fft_out_Y[2*k+1];
            float32_t rz=fft_out_Z[2*k], iz=fft_out_Z[2*k+1];
            magX = sqrtf(rx*rx + ix*ix);
            magY = sqrtf(ry*ry + iy*iy);
            magZ = sqrtf(rz*rz + iz*iz);
        }
        float32_t combined = sqrtf(magX*magX + magY*magY + magZ*magZ);
        float32_t scale = (k==0||k==FFT_SIZE/2) ? (1.0f/FFT_SIZE) : (2.0f/FFT_SIZE);
        magnitude_total[k] = combined * scale;
    }
}


// -----------------------------------------------------------------
// show_fft_result(): Find the peak frequency in 0.5–10 Hz band
//   above a noise floor, print it, and update shared variables.
// -----------------------------------------------------------------
void show_fft_result() {
    const float  res        = SAMPLE_RATE / FFT_SIZE;
    const float  noise_floor= 0.002f;       // tune experimentally
    const float  min_hz     = 0.5f,
                 max_hz     = 10.0f;
    uint32_t     min_bin    = ceil(min_hz / res);
    uint32_t     max_bin    = floor(max_hz / res);

    float  bestVal = 0;
    uint32_t bestIdx = 0;
    for(uint32_t k = min_bin; k <= max_bin; k++){
        if(magnitude_total[k] > bestVal) {
            bestVal = magnitude_total[k];
            bestIdx = k;
        }
    }

    float peakHz = 0;
    if(bestVal >= noise_floor) {
        peakHz = bestIdx * res;
    }

    printf("\r\n=== FFT Result ===\r\n");
    if(peakHz > 0) {
        printf("Peak at %.2f Hz (Mag: %.3f g)\r\n", peakHz, bestVal);
    } else {
        printf("No significant peak (noise floor: %.3f g)\r\n", noise_floor);
    }
    printf("===================\r\n");

    sharedPeakHz.store(peakHz);
    sharedMaxVal.store(bestVal);
}

// -----------------------------------------------------------------
// main(): Initialization and main loop.
// -----------------------------------------------------------------
int main() {
    // start blink thread
    blinkThread.start(callback(blink_task));

    // I2C init & WHO_AM_I check
    i2c.frequency(400000);
    uint8_t id = read_register(WHO_AM_I);
    printf("WHO_AM_I = 0x%02X (Expected: 0x6A)\r\n", id);
    if(id != 0x6A) {
        printf("Error: LSM6DSL not found!\r\n");
        while(1);
    }

    // configure accel: 416 Hz, ±2 g
    write_register(CTRL1_XL, 0x60);
    printf("Accelerometer configured: %d Hz, ±2 g\r\n", SAMPLE_RATE_HZ);

    // button to toggle sampling
    button.fall(&button_pressed);

    // FFT init
    if(arm_rfft_fast_init_f32(&FFT_Instance, FFT_SIZE) != ARM_MATH_SUCCESS){
        printf("FFT init failed!\r\n");
        while(1);
    }

    printf("Press button to Start/Stop Data Collection\r\n");

    while(true) {
        if(buttonPressed) {
            buttonPressed = false;
            isReading     = !isReading;
            led3          = isReading;
            record_count  = 0;
            printf("%s data collection\r\n",
                   isReading ? "Started" : "Stopped");
        }

        if(isReading) {
            if(record_count < RECORD_SIZE) {
                // wait for data-ready
                while(!(read_register(STATUS_REG) & 0x01));
                int16_t ax = read_16bit_value(OUTX_L_XL, OUTX_H_XL);
                int16_t ay = read_16bit_value(OUTY_L_XL, OUTY_H_XL);
                int16_t az = read_16bit_value(OUTZ_L_XL, OUTZ_H_XL);

                const float32_t sens = 0.061f / 1000.0f;
                record_X[record_count] = ax * sens;
                record_Y[record_count] = ay * sens;
                record_Z[record_count] = az * sens;
                record_count++;
            }
            else {
                printf("Recorded %u samples over %d seconds.\r\n",
                       RECORD_SIZE, RECORD_DURATION_S);
                // copy first FFT_SIZE samples
                for(int i=0;i<FFT_SIZE;i++){
                    sample_data_X[i] = record_X[i];
                    sample_data_Y[i] = record_Y[i];
                    sample_data_Z[i] = record_Z[i];
                }
                run_fft_all();
                show_fft_result();

                // prep next
                record_count = 0;
                ThisThread::sleep_for(500ms);
                printf("Next batch...\r\n");
            }
        }
        else {
            ThisThread::sleep_for(100ms);
        }
    }
} 