// Root tab layout for live perception and calibration.

import SwiftUI

struct ContentView: View {
    var body: some View {
        TabView {
            LiveView()
                .tabItem { Label("Live", systemImage: "car.side") }
            CalibrationView()
                .tabItem { Label("Calibrate", systemImage: "level") }
        }
    }
}

#Preview {
    ContentView()
}
