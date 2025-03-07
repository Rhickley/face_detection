import 'dart:async';
import 'dart:isolate';
import 'dart:ui';
import 'dart:typed_data';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

late List<CameraDescription> _cameras;

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  _cameras = await availableCameras();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Otimizado Face Detection',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'Face Detection'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> with WidgetsBindingObserver {
  late CameraController controller;
  DateTime? lastProcessedTime;
  bool faceDetected = false;
  bool isProcessing = false;

  // Comunicação entre isolates
  ReceivePort? _receivePort;
  Isolate? _isolate;
  SendPort? _sendPort;

  // Orientação do dispositivo
  Orientation? _currentOrientation;

  // Sinaliza quando o isolate está pronto
  final Completer<void> _isolateReady = Completer();

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this); // Observa mudanças de orientação
    _currentOrientation = _calculateCurrentOrientation();
    _initIsolate();// Inicializa o isolate de processamento
    _initializeCamera();
  }

  // Calcula orientação com base nas dimensões da tela
  Orientation _calculateCurrentOrientation() {
    return WidgetsBinding.instance.window.physicalSize.aspectRatio > 1
        ? Orientation.landscape
        : Orientation.portrait;
  }

  @override
  void didChangeMetrics() {
    super.didChangeMetrics();
    setState(() {
      _currentOrientation = _calculateCurrentOrientation(); // Atualiza orientação
    });
  }

  void _initializeCamera() {
    controller = CameraController(_cameras[1], ResolutionPreset.high);
    controller.initialize().then((_) {
      if (!mounted) return;
      controller.startImageStream(_processCameraImage);
      setState(() {});
    }).catchError((e) {
      debugPrint("Camera error: ${e.toString()}");
    });
  }

  // Configura o isolate paralelo
  Future<void> _initIsolate() async {
    _receivePort = ReceivePort();

    // Spawn do novo isolate
    _isolate = await Isolate.spawn(
      _isolateEntry,
      _IsolateInitializationParams(
        mainSendPort: _receivePort!.sendPort,
        rootIsolateToken: RootIsolateToken.instance!,
      ),
      debugName: 'FaceDetectionIsolate',
    );

    // Escuta mensagens do isolate
    _receivePort!.listen((message) {
      if (message is SendPort) {
        _sendPort = message;
        _isolateReady.complete();
      } else if (message is bool) {
        setState(() => faceDetected = message);// Atualiza estado de detecção
      }
    });
  }

  // Ponto de entrada do isolate
  static void _isolateEntry(_IsolateInitializationParams params) {
    BackgroundIsolateBinaryMessenger.ensureInitialized(params.rootIsolateToken);

    final faceDetector = FaceDetector(
      options: FaceDetectorOptions(
        enableContours: false,
        enableLandmarks: true, // Habilita marcos faciais
        enableClassification: true, // Habilita classificações (sorriso, olhos)
        performanceMode: FaceDetectorMode.accurate,
        enableTracking: true,
      ),
    );

    final receivePort = ReceivePort();
    params.mainSendPort.send(receivePort.sendPort);

    // Processa cada frame recebido
    receivePort.listen((message) async {
      if (message is List<dynamic>) {
        final SendPort replyPort = message[0];
        final CameraImage image = message[1];

        try {
          final bytes = await _convertImageToBytes(image);
          final inputImage = InputImage.fromBytes(
            bytes: bytes,
            metadata: InputImageMetadata(
              size: Size(image.width.toDouble(), image.height.toDouble()),
              rotation: _getRotationFromCamera(params.orientation),
              format: _getImageFormat(),
              bytesPerRow: image.planes[0].bytesPerRow,
            ),
          );

          final faces = await faceDetector.processImage(inputImage);
          replyPort.send(faces.isNotEmpty);
        } catch (e) {
          replyPort.send(false);
        }
      }
    });
  }

  static Future<Uint8List> _convertImageToBytes(CameraImage image) async {
    final WriteBuffer buffer = WriteBuffer();
    for (final plane in image.planes) {
      buffer.putUint8List(plane.bytes);
    }
    return buffer.done().buffer.asUint8List();
  }

  static InputImageRotation _getRotationFromCamera(Orientation? orientation) {
    if (orientation == Orientation.portrait) {
      return Platform.isAndroid
          ? InputImageRotation.rotation90deg
          : InputImageRotation.rotation0deg;
    }
    return InputImageRotation.rotation0deg;
  }

  static InputImageFormat _getImageFormat() {
    return defaultTargetPlatform == TargetPlatform.android
        ? InputImageFormat.nv21
        : InputImageFormat.yuv420;
  }

  void _processCameraImage(CameraImage image) async {
    if (lastProcessedTime != null &&
        DateTime.now().difference(lastProcessedTime!).inMilliseconds < 500) return;
    if (isProcessing || !_isolateReady.isCompleted) return;

    isProcessing = true;
    lastProcessedTime = DateTime.now();

    final replyPort = ReceivePort();
    _sendPort!.send([replyPort.sendPort, image]);// Envia frame para isolate

    replyPort.listen((message) {
      if (message is bool) {
        setState(() => faceDetected = message);
      }
      replyPort.close();
      isProcessing = false;
    });
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _isolate?.kill(priority: Isolate.immediate);// Encerra isolate
    _receivePort?.close();
    controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
      ),
      body: Stack(
        children: [
          CameraPreview(controller),
          _buildDetectionIndicator(),
        ],
      ),
    );
  }

  Widget _buildDetectionIndicator() {
    return Positioned(
      bottom: 50,
      left: 0,
      right: 0,
      child: Center(
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: faceDetected ? Colors.green : Colors.red,
            borderRadius: BorderRadius.circular(8),
          ),
          child: Text(
            faceDetected ? "Rosto Detectado!" : "Procurando...",
            style: const TextStyle(color: Colors.white),
          ),
        ),
      ),
    );
  }
}

// Parâmetros para inicialização do isolate
class _IsolateInitializationParams {
  final SendPort mainSendPort;
  final RootIsolateToken rootIsolateToken;
  final Orientation? orientation;

  const _IsolateInitializationParams({
    required this.mainSendPort,
    required this.rootIsolateToken,
    this.orientation,
  });
}