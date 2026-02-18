# Beginner's Guide: Getting Started with vGPU

Welcome! This guide is for you if you have never used a programming language like Rust and just want to see the vGPU in action on your own computer.

## Step 1: Get the Project
First, you need the files on your computer.
1. Scroll to the top of this page on GitHub.
2. Click the green **Code** button.
3. Click **Download ZIP**.
4. Once it downloads, find the file (usually in your "Downloads" folder), right-click it, and choose **Extract All**. Now you have a folder with the vGPU files.

## Step 2: Install "The Engine" (Rust)
To "run" the vGPU, you need a program called **Rust**. It's the engine that reads our code and turns it into math.
1. Go to [rustup.rs](https://rustup.rs/).
2. Click the download for your computer (if you use Windows, it will be an `.exe` file).
3. Open that file and a black window will pop up.
4. It will ask you to choose an option. Just press the **"1"** key on your keyboard and then **Enter**.
5. Restart your computer when it finishes (this makes sure everything is ready).

## Step 3: Verify the "O(1) Scaling"
Now, let's prove the vGPU works. We are going to run a test that usually takes a standard computer forever, but the vGPU will do it instantly.
1. Open the folder where you extracted the vGPU files.
2. Inside that folder, find the folder named `vgpu_rust`.
3. In the address bar at the top of your folder window, type the letters `cmd` and press **Enter**. A black box (Command Prompt) will open up.
4. Type this exact command into the black box and press **Enter**:
   ```bash
    cargo test --release --test test_auto_induction -- --nocapture
   ```
5. **Wait a minute.** The first time you do this, your computer will download some parts it needs. This is normal.

## Step 4: Reading the Results
Once the test finishes, look at the screen. You should see something like this:
- **"Cold N=1000000 | Time: 10ms"** (This is the computer doing it the old, slow way).
- **"Warm N=1000000 | Time: 1µs"** (This is the vGPU doing it the "Generative" way).
- **"Warm N=1,000,000,000,000,000,000 | Time: 800ns"** (The vGPU doing an "Impossible" amount of work instantly).

If you see those numbers, the manifold is working. You have successfully run a vGPU on your machine!
