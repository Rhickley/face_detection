import 'dart:async';
import 'dart:isolate';
import 'dart:ui';
import 'dart:typed_data';
import 'dart:io';
import 'dart:math' as math;
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

late List<CameraDescription> _cameras;

// Configurações do modelo
const facialModel = 'assets/modelo_convertido.tflite';
const inputSize = 48;
const facialLabel = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'];

const facialLabelTraduzidas = {
  'Angry': 'Raiva',
  'Disgust': 'Desgosto',
  'Fear': 'Medo',
  'Happy': 'Feliz',
  'Neutral': 'Neutro',
  'Sad': 'Triste',
  'Surprise': 'Surpresa'
};

// Classe para representar a região do rosto detectado
class FaceRegion {
  final int x;
  final int y;
  final int width;
  final int height;

  FaceRegion(this.x, this.y, this.width, this.height);
}

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
      title: 'Detector de Emoções',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'Detecção de Emoções'),
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
  CameraController? controller;
  DateTime? lastProcessedTime;
  bool faceDetected = false;
  bool isProcessing = false;
  String detectedEmotion = "Procurando...";
  bool isFrontCamera = true;

  // Isolate para detecção de rostos
  ReceivePort? _receivePort;
  Isolate? _isolate;
  SendPort? _sendPort;
  Orientation? _currentOrientation;
  final Completer<void> _isolateReady = Completer<void>();

  // Classificador de emoções
  late Interpreter _interpreter;
  List<int>? _inputShape;
  List<int>? _outputShape;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _currentOrientation = _calculateCurrentOrientation();
    _initIsolate();
    _loadModel().then((_) => _initializeCamera(1));
    debugPrint("🔄 Inicializando aplicação...");
  }

  Future<void> _loadModel() async {
    try {
      debugPrint("⬇️ Iniciando carregamento do modelo...");
      final options = InterpreterOptions();

      // Habilitar XNNPACK para melhor performance
      try {
        if (Platform.isAndroid) {
          debugPrint("⚡ Tentando habilitar XNNPACK delegate...");
          options.addDelegate(XNNPackDelegate());
          debugPrint("✅ XNNPACK delegate habilitado com sucesso");
        }
      } catch (e) {
        debugPrint("⚠️ Falha ao habilitar XNNPACK: $e");
      }

      debugPrint("📦 Carregando modelo TFLite...");
      _interpreter = await Interpreter.fromAsset(facialModel, options: options);
      debugPrint("✅ Modelo carregado com sucesso");

      // Obter shapes de entrada e saída
      debugPrint("📊 Obtendo tensores de entrada/saída...");
      final inputTensors = _interpreter.getInputTensors();
      final outputTensors = _interpreter.getOutputTensors();

      if (inputTensors.isNotEmpty) {
        _inputShape = inputTensors.first.shape;
        debugPrint("📥 Input shape: $_inputShape");
      } else {
        debugPrint("⚠️ Nenhum tensor de entrada encontrado");
      }

      if (outputTensors.isNotEmpty) {
        _outputShape = outputTensors.first.shape;
        debugPrint("📤 Output shape: $_outputShape");
      } else {
        debugPrint("⚠️ Nenhum tensor de saída encontrado");
      }

      // Verificar se o shape de entrada é compatível
      if (_inputShape == null || _inputShape!.length != 4 ||
          _inputShape![1] != inputSize ||
          _inputShape![2] != inputSize ||
          _inputShape![3] != 1) {
        debugPrint("⚠️ Modelo espera um input de [1, $inputSize, $inputSize, 1] mas recebeu $_inputShape");
      } else {
        debugPrint("🆗 Shape de entrada compatível");
      }
    } catch (e) {
      debugPrint("❌ Erro crítico ao carregar modelo: $e");
      rethrow;
    }
  }

  Future<void> _toggleCamera() async {
    debugPrint("🔄 Alternando câmera...");
    isFrontCamera = !isFrontCamera;
    await _initializeCamera(isFrontCamera ? 1 : 0);
  }

  Orientation _calculateCurrentOrientation() {
    final orientation = WidgetsBinding.instance.window.physicalSize.aspectRatio > 1
        ? Orientation.landscape
        : Orientation.portrait;
    debugPrint("🧭 Orientação calculada: $orientation");
    return orientation;
  }

  @override
  void didChangeMetrics() {
    super.didChangeMetrics();
    final newOrientation = _calculateCurrentOrientation();
    debugPrint("🔄 Mudança de orientação detectada: $_currentOrientation → $newOrientation");
    setState(() {
      _currentOrientation = newOrientation;
    });
  }

  Future<void> _initializeCamera(int cameraIndex) async {
    try {
      debugPrint("📷 Inicializando câmera $cameraIndex...");
      if (controller != null) {
        debugPrint("♻️ Liberando câmera anterior...");
        await controller!.dispose();
      }

      controller = CameraController(_cameras[cameraIndex], ResolutionPreset.high);
      debugPrint("⚙️ Configurando câmera...");
      await controller!.initialize();

      if (mounted) {
        debugPrint("🎥 Iniciando stream de imagens...");
        controller!.startImageStream(_processCameraImage);
        setState(() {});
        debugPrint("✅ Câmera ${isFrontCamera ? "frontal" : "traseira"} inicializada com sucesso");
      }
    } catch (e) {
      debugPrint("❌ Falha na inicialização da câmera: ${e.toString()}");
      setState(() => controller = null);
    }
  }

  Future<void> _initIsolate() async {
    debugPrint("🧵 Inicializando isolate para detecção de rostos...");
    _receivePort = ReceivePort();
    _isolate = await Isolate.spawn(
      _isolateEntry,
      _IsolateInitializationParams(
        mainSendPort: _receivePort!.sendPort,
        rootIsolateToken: RootIsolateToken.instance!,
      ),
      debugName: 'FaceDetectionIsolate',
    );

    _receivePort!.listen((message) {
      if (message is SendPort) {
        debugPrint("📨 Porta de comunicação do isolate recebida");
        _sendPort = message;
        _isolateReady.complete();
        debugPrint("✅ Isolate de detecção de rostos inicializado com sucesso");
      } else if (message is FaceRegion) {
        debugPrint("👤 Rosto detectado em ${message.x},${message.y} ${message.width}x${message.height}");
        setState(() => faceDetected = true);
      } else if (message == null) {
        debugPrint("❌ Nenhum rosto detectado");
        setState(() => faceDetected = false);
      }
    });
  }

  static void _isolateEntry(_IsolateInitializationParams params) {
    debugPrint("🧵 Isolate de detecção iniciado");
    BackgroundIsolateBinaryMessenger.ensureInitialized(params.rootIsolateToken);
    final faceDetector = FaceDetector(
      options: FaceDetectorOptions(
        enableContours: true,
        enableLandmarks: true,
        enableClassification: false,
        performanceMode: FaceDetectorMode.accurate,
        enableTracking: true,
        minFaceSize: 0.3,
      ),
    );

    final receivePort = ReceivePort();
    params.mainSendPort.send(receivePort.sendPort);

    receivePort.listen((message) async {
      if (message is List<dynamic>) {
        debugPrint("📷 Processando imagem no isolate...");
        final SendPort replyPort = message[0];
        final Uint8List nv21Bytes = message[1];
        final int width = message[2];
        final int height = message[3];
        final Orientation orientation = message[4];

        try {
          final inputImage = InputImage.fromBytes(
            bytes: nv21Bytes,
            metadata: InputImageMetadata(
              size: Size(width.toDouble(), height.toDouble()),
              rotation: _getRotationFromCamera(orientation),
              format: InputImageFormat.nv21,
              bytesPerRow: width,
            ),
          );

          debugPrint("🔍 Detectando rostos...");
          final faces = await faceDetector.processImage(inputImage);

          if (faces.isNotEmpty) {
            // Pegar o primeiro rosto (mais proeminente)
            final face = faces.first;
            final rect = face.boundingBox;

            // Ajustar coordenadas para garantir que estão dentro dos limites
            final left = rect.left.clamp(0, width.toDouble());
            final top = rect.top.clamp(0, height.toDouble());
            final right = rect.right.clamp(0, width.toDouble());
            final bottom = rect.bottom.clamp(0, height.toDouble());

            // Criar região do rosto
            final faceRegion = FaceRegion(
              left.toInt(),
              top.toInt(),
              (right - left).toInt(),
              (bottom - top).toInt(),
            );

            debugPrint("✅ Rosto detectado em $left,$top ${faceRegion.width}x${faceRegion.height}");
            replyPort.send(faceRegion);
          } else {
            debugPrint("❌ Nenhum rosto encontrado");
            replyPort.send(null);
          }
        } catch (e) {
          debugPrint("❌ Erro no isolate durante detecção: $e");
          replyPort.send(null);
        }
      }
    });
  }

  Uint8List _yuv420ToNv21(CameraImage image) {
    debugPrint("🔄 Convertendo YUV420 para NV21...");
    final Uint8List yBuffer = image.planes[0].bytes;
    final Uint8List uBuffer = image.planes[1].bytes;
    final Uint8List vBuffer = image.planes[2].bytes;

    final int ySize = yBuffer.length;
    final int uvSize = (image.width ~/ 2) * (image.height ~/ 2);

    final Uint8List nv21 = Uint8List(ySize + uvSize * 2);
    nv21.setRange(0, ySize, yBuffer);

    int index = ySize;
    for (int i = 0; i < uvSize; i++) {
      nv21[index++] = vBuffer[i];
      nv21[index++] = uBuffer[i];
    }

    debugPrint("✅ Conversão YUV→NV21 concluída");
    return nv21;
  }

  Float32List _yuvToModelInput(CameraImage image, FaceRegion faceRegion) {
    final inputBuffer = Float32List(1 * inputSize * inputSize * 1);
    final yPlane = image.planes[0].bytes;
    final uPlane = image.planes[1].bytes;
    final vPlane = image.planes[2].bytes;
    final yRowStride = image.planes[0].bytesPerRow;
    final uvRowStride = image.planes[1].bytesPerRow;
    final uvPixelStride = image.planes[1].bytesPerPixel;

    // Criar uma imagem temporária para o rosto recortado
    final debugImage = img.Image(width: inputSize, height: inputSize);

    // Fatores de escala para redimensionar o rosto para inputSize
    final scaleX = faceRegion.width / inputSize;
    final scaleY = faceRegion.height / inputSize;

    // Determinar se precisa rotacionar (para portrait)
    final bool needsRotation = _currentOrientation == Orientation.portrait;
    final bool isFrontCameraRotated = isFrontCamera && needsRotation;

    int pixelIndex = 0;
    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        // Calcular coordenadas na região do rosto
        int faceX = (x * scaleX).toInt();
        int faceY = (y * scaleY).toInt();

        // Coordenadas na imagem original
        int origX = faceRegion.x + faceX;
        int origY = faceRegion.y + faceY;

        // Aplicar rotação de 90 graus no sentido horário se necessário
        if (needsRotation) {
          final temp = origX;
          origX = origY;
          origY = image.height - 1 - temp;
        }

        // Espelhar horizontalmente se for câmera frontal (após a rotação)
        if (isFrontCameraRotated) {
          origX = image.width - 1 - origX;
        }

        // Garantir que as coordenadas estão dentro dos limites
        origX = origX.clamp(0, image.width - 1);
        origY = origY.clamp(0, image.height - 1);

        // Obter valor Y (luminância)
        final pixelValue = yPlane[origY * yRowStride + origX];

        // Normalizar para [0,1]
        inputBuffer[pixelIndex++] = pixelValue / 255.0;

        // Debug: Armazenar na imagem visualizável
        debugImage.setPixelRgb(x, y, pixelValue, pixelValue, pixelValue);
      }
    }

    _saveDebugImage(debugImage);
    return inputBuffer;
  }

  void _saveDebugImage(img.Image image) async {
    try {
      final directory = Platform.isAndroid
          ? Directory('/storage/emulated/0/Download') // Android
          : await getApplicationDocumentsDirectory(); // iOS
      final file = File('${directory.path}/debug_emotion_input.png');
      await file.writeAsBytes(img.encodePng(image));
      debugPrint('🖼️ Imagem salva em: ${file.path}');
    } catch (e) {
      debugPrint('❌ Erro ao salvar imagem: $e');
    }
  }

  static InputImageRotation _getRotationFromCamera(Orientation orientation) {
    final rotation = orientation == Orientation.portrait
        ? Platform.isAndroid
        ? InputImageRotation.rotation270deg
        : InputImageRotation.rotation180deg
        : InputImageRotation.rotation0deg;
    debugPrint("🔄 Rotação da câmera: $rotation");
    return rotation;
  }

  void _processCameraImage(CameraImage image) async {
    final now = DateTime.now();
    if (lastProcessedTime != null && now.difference(lastProcessedTime!).inMilliseconds < 500) {
      debugPrint("⏭️ Pulando frame - processamento muito rápido");
      return;
    }
    if (isProcessing) {
      debugPrint("⏭️ Pulando frame - já em processamento");
      return;
    }
    if (!_isolateReady.isCompleted) {
      debugPrint("⏭️ Pulando frame - isolate não pronto");
      return;
    }
    if (_interpreter == null) {
      debugPrint("⏭️ Pulando frame - modelo não carregado");
      return;
    }

    isProcessing = true;
    lastProcessedTime = now;
    debugPrint("\n🔄 Processando novo frame (${image.width}x${image.height})...");

    try {
      final replyPort = ReceivePort();
      debugPrint("📨 Enviando imagem para o isolate...");
      final nv21Bytes = _yuv420ToNv21(image);
      _sendPort!.send([
        replyPort.sendPort,
        nv21Bytes,
        image.width,
        image.height,
        _currentOrientation!,
      ]);

      replyPort.listen((message) async {
        if (message is FaceRegion) {
          debugPrint("👤 Rosto detectado no frame em ${message.x},${message.y} ${message.width}x${message.height}");
          setState(() => faceDetected = true);

          debugPrint("🧠 Preparando para análise de emoções...");
          // Conversão YUV para input do modelo com recorte do rosto
          final input = _yuvToModelInput(image, message);

          // Verificar se temos os shapes necessários
          if (_inputShape == null || _outputShape == null) {
            debugPrint("⚠️ Shapes de input/output não disponíveis");
            setState(() => detectedEmotion = "Erro no modelo");
            return;
          }

          // Preparar buffer de saída
          final outputSize = _outputShape!.reduce((a, b) => a * b);
          final output = List.filled(outputSize, 0.0).reshape(_outputShape!);

          try {
            debugPrint("⚡ Executando inferência do modelo...");
            _interpreter.run(input.reshape(_inputShape!), output);

            debugPrint("📊 Saídas do modelo: ${output[0]}");

            // Encontrar emoção com maior confiança
            int maxIndex = 0;
            double maxConfidence = output[0][0];
            for (int i = 1; i < output[0].length; i++) {
              if (output[0][i] > maxConfidence) {
                maxConfidence = output[0][i];
                maxIndex = i;
              }
            }

            String emotion = facialLabel[maxIndex];
            String emotionTranslated = facialLabelTraduzidas[emotion] ?? 'Desconhecido';
            String confidence = (maxConfidence * 100).toStringAsFixed(1);

            debugPrint("🎭 Emoção detectada: $emotionTranslated ($confidence%)");
            setState(() => detectedEmotion = "$emotionTranslated ($confidence%)");
          } catch (e) {
            debugPrint("❌ Erro durante inferência: $e");
            setState(() => detectedEmotion = "Erro na detecção");
          }
        } else {
          debugPrint("👀 Nenhum rosto - resetando emoção");
          setState(() {
            faceDetected = false;
            detectedEmotion = "Procurando...";
          });
        }
        replyPort.close();
        isProcessing = false;
        debugPrint("✅ Processamento do frame concluído");
      });
    } catch (e) {
      debugPrint("❌ Erro crítico no processamento: $e");
      isProcessing = false;
    }
  }

  @override
  void dispose() {
    debugPrint("♻️ Liberando recursos...");
    WidgetsBinding.instance.removeObserver(this);
    _interpreter.close();
    _isolate?.kill(priority: Isolate.immediate);
    _receivePort?.close();
    controller?.dispose();
    super.dispose();
    debugPrint("✅ Recursos liberados");
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
      ),
      body: Stack(
        children: [
          if (controller != null && controller!.value.isInitialized)
            CameraPreview(controller!),
          _buildDetectionIndicator(),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _toggleCamera,
        tooltip: 'Alternar câmera',
        child: const Icon(Icons.switch_camera),
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
            faceDetected ? "Emoção: $detectedEmotion" : "Procurando rosto...",
            style: const TextStyle(color: Colors.white, fontSize: 18),
          ),
        ),
      ),
    );
  }
}

class _IsolateInitializationParams {
  final SendPort mainSendPort;
  final RootIsolateToken rootIsolateToken;
  const _IsolateInitializationParams({
    required this.mainSendPort,
    required this.rootIsolateToken,
  });
}