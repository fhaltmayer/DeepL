//
//  CameraView.swift
//  Vangogh_StoryBoard
//
//  Created by Filip Haltmayer on 10/31/20.
//

import UIKit
import AVFoundation

class CameraView: UIViewController {

    private let captureSession = AVCaptureSession()
    
    private lazy var previewLayer: AVCaptureVideoPreviewLayer = {
        let preview = AVCaptureVideoPreviewLayer(session: self.captureSession)
        preview.videoGravity = .resizeAspect
        return preview
    }()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        
        case .authorized:
            break
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) {
                granted in if granted {
                    
                }
            }
        case .denied:
            return
            
        case .restricted:
            return
            
        @unknown default:
            print("Something broke with permissions")
            return
        }
        self.setResolution()
        self.addCameraInput()
        self.addPreviewLayer()
        self.captureSession.startRunning()
    }
   
    
    private func setResolution() {
        print(ViewController.myString)
        switch ViewController.myString{
        case "192x144":
            self.captureSession.sessionPreset = AVCaptureSession.Preset.low //192x144
        case "352x288":
            self.captureSession.sessionPreset = AVCaptureSession.Preset.cif352x288
        case "480x360":
            self.captureSession.sessionPreset = AVCaptureSession.Preset.medium //480x360
        case "640x480":
            self.captureSession.sessionPreset = AVCaptureSession.Preset.vga640x480
        case "1280x720":
            self.captureSession.sessionPreset = AVCaptureSession.Preset.hd1280x720
        default:
            print("Failed to pick res")
            return
        }
        
        
        
        
        
        
        
        
    }
    
    private func addCameraInput() {
        let devices = AVCaptureDevice.DiscoverySession.init(deviceTypes: [AVCaptureDevice.DeviceType.builtInWideAngleCamera], mediaType: AVMediaType.video, position: AVCaptureDevice.Position.back).devices
        do {
            if let captureDevice = devices.first {
                captureSession.addInput(try AVCaptureDeviceInput(device: captureDevice))
            }
        }
        catch {
            print("Failed to set CaptureDeviceInput")
        }
    }
    
    private func addPreviewLayer() {
        self.view.layer.addSublayer(self.previewLayer)
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        let cur
        self.previewLayer.frame = self.view.bounds
    }
    

}

