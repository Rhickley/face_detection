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

class ModelConfig {
  final String path;
  final int inputSize;
  final bool isColor;
  final List<String> labels;
  final Map<String, String> labelTranslations;

  ModelConfig({
    required this.path,
    required this.inputSize,
    required this.isColor,
    required this.labels,
    required this.labelTranslations,
  });
}

final Map<String, ModelConfig> modelConfigs = {
  'Modelo 1': ModelConfig(
    path: 'assets/modelo_convertido.tflite',
    inputSize: 48,
    isColor: false,
    labels: ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'],
    labelTranslations: {
      'Angry': 'Raiva',
      'Disgust': 'Desgosto',
      'Fear': 'Medo',
      'Happy': 'Feliz',
      'Neutral': 'Neutro',
      'Sad': 'Triste',
      'Surprise': 'Surpresa'
    },
  ),
  'Modelo 2': ModelConfig(
    path: 'assets/model.tflite',
    inputSize: 224,
    isColor: true,
    labels: ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised'],
    labelTranslations: {
      'Angry': 'Raiva',
      'Fear': 'Medo',
      'Happy': 'Feliz',
      'Neutral': 'Neutro',
      'Sad': 'Triste',
      'Surprised': 'Surpresa'
    },
  ),
  'Modelo 3': ModelConfig(
    path: 'assets/fe93.tflite',
    inputSize: 224,
    isColor: true,
    labels: ['驚喜', '害怕', '噁心', '開心', '傷心', '生氣', '無'],
    labelTranslations: {
      '驚喜': 'Surpresa',
      '害怕': 'Medo',
      '噁心': 'Nojo',
      '開心': 'Feliz',
      '傷心': 'Triste',
      '生氣': 'Raiva',
      '無': 'Neutro'
    },
  ),
};

class FaceRegion {
  final Rect rect;
  final Map<FaceLandmarkType, FaceLandmark>? landmarks;

  FaceRegion(this.rect, [this.landmarks]);
}

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  debugPrint("📷 Obtendo câmeras disponíveis...");
  _cameras = await availableCameras();
  debugPrint("✅ Câmeras obtidas: ${_cameras.length} encontradas");
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    debugPrint("🖼️ Construindo widget MyApp...");
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
  FaceRegion? currentFaceRegion;
  String selectedModel = 'Modelo 1';

  ReceivePort? _receivePort;
  Isolate? _isolate;
  SendPort? _sendPort;
  Orientation? _currentOrientation;
  final Completer<void> _isolateReady = Completer<void>();

  late Interpreter _interpreter;
  List<int>? _inputShape;
  List<int>? _outputShape;
  int _inputSize = 48;
  bool _isColorInput = false;

  bool _saveDebugImages = false;

  @override
  void initState() {
    super.initState();
    debugPrint("🔄 Inicializando estado...");
    WidgetsBinding.instance.addObserver(this);
    _currentOrientation = _calculateCurrentOrientation();
    _initIsolate();
    _loadModel().then((_) => _initializeCamera(1));
  }

  Future<void> _loadModel() async {
    try {
      debugPrint("⬇️ Iniciando carregamento do modelo $selectedModel...");
      final modelConfig = modelConfigs[selectedModel]!;
      final options = InterpreterOptions();

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
      _interpreter = await Interpreter.fromAsset(modelConfig.path, options: options);
      debugPrint("✅ Modelo carregado com sucesso");

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

      _inputSize = modelConfig.inputSize;
      _isColorInput = modelConfig.isColor;

      if (_inputShape == null || _inputShape!.length != 4 ||
          _inputShape![1] != _inputSize ||
          _inputShape![2] != _inputSize ||
          _inputShape![3] != (_isColorInput ? 3 : 1)) {
        debugPrint(
            "⚠️ Modelo espera um input de [1, $_inputSize, $_inputSize, ${_isColorInput ? 3 : 1}] mas recebeu $_inputShape");
      } else {
        debugPrint("🆗 Shape de entrada compatível");
      }
    } catch (e) {
      debugPrint("❌ Erro crítico ao carregar modelo: $e");
      rethrow;
    }
  }

  Future<void> _changeModel(String newModel) async {
    if (selectedModel == newModel) return;

    debugPrint("🔄 Alterando modelo de $selectedModel para $newModel...");
    setState(() {
      selectedModel = newModel;
      detectedEmotion = "Carregando...";
    });

    _interpreter.close();

    await _loadModel();

    setState(() {
      detectedEmotion = "Procurando...";
    });
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
        debugPrint("👤 Rosto detectado no frame - Posição: ${message.rect}");
        setState(() {
          faceDetected = true;
          currentFaceRegion = message;
        });
      } else if (message == null) {
        debugPrint("❌ Nenhum rosto detectado");
        setState(() {
          faceDetected = false;
          currentFaceRegion = null;
        });
      }
    });
  }

  static void _isolateEntry(_IsolateInitializationParams params) {
    debugPrint("🧵 Isolate de detecção iniciado");
    BackgroundIsolateBinaryMessenger.ensureInitialized(params.rootIsolateToken);
    final faceDetector = FaceDetector(
      options: FaceDetectorOptions(
        enableContours: false,
        enableLandmarks: false,
        enableClassification: false,
        performanceMode: FaceDetectorMode.fast,
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
            debugPrint("✅ ${faces.length} rosto(s) detectado(s)");
            final face = faces.first;
            final rect = face.boundingBox;

            Map<FaceLandmarkType, FaceLandmark>? landmarksMap;
            if (face.landmarks != null && face.landmarks!.isNotEmpty) {
              landmarksMap = {};
              for (final entry in face.landmarks!.entries) {
                if (entry.value != null) {
                  landmarksMap[entry.key] = entry.value!;
                }
              }
            }

            replyPort.send(FaceRegion(
              rect,
              landmarksMap?.isNotEmpty == true ? landmarksMap : null,
            ));
          } else {
            debugPrint("❌ Nenhum rosto encontrado na imagem");
            replyPort.send(null);
          }
        } catch (e) {
          debugPrint("❌ Erro no isolate durante detecção: $e");
          replyPort.send(null);
        }
      }
    });
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

  void _saveDebugImage(img.Image image) async {
    try {
      debugPrint("💾 Tentando salvar imagem de debug...");
      final directory = Platform.isAndroid
          ? Directory('/storage/emulated/0/Download')
          : await getApplicationDocumentsDirectory();

      final timestamp = DateTime.now().millisecondsSinceEpoch;
      final modelName = selectedModel.replaceAll(' ', '_').toLowerCase();
      final file = File('${directory.path}/debug_${modelName}_$timestamp.png');

      await file.writeAsBytes(img.encodePng(image));
      debugPrint('🖼️ Imagem salva em: ${file.path}');
    } catch (e) {
      debugPrint('❌ Erro ao salvar imagem: $e');
    }
  }

  Float32List _yuvToModelInput(CameraImage image, FaceRegion faceRegion) {
    debugPrint("🔄 Preparando input para o modelo $selectedModel...");
    final modelConfig = modelConfigs[selectedModel]!;
    final channels = modelConfig.isColor ? 3 : 1;
    final inputBuffer = Float32List(1 * _inputSize * _inputSize * channels);

    final yPlane = image.planes[0].bytes;
    final yRowStride = image.planes[0].bytesPerRow;
    final uvPlane = image.planes[1].bytes;
    final uvRowStride = image.planes[1].bytesPerRow;
    final uvPixelStride = image.planes[1].bytesPerPixel!;

    final debugImage = img.Image(width: _inputSize, height: _inputSize);

    final scaleX = faceRegion.rect.width / _inputSize;
    final scaleY = faceRegion.rect.height / _inputSize;

    final bool needsRotation = _currentOrientation == Orientation.portrait;
    final bool isFrontCameraRotated = isFrontCamera && needsRotation;

    int pixelIndex = 0;
    for (int y = 0; y < _inputSize; y++) {
      for (int x = 0; x < _inputSize; x++) {
        int faceX = (x * scaleX).toInt();
        int faceY = (y * scaleY).toInt();

        int origX = faceRegion.rect.left.toInt() + faceX;
        int origY = faceRegion.rect.top.toInt() + faceY;

        if (needsRotation) {
          final temp = origX;
          origX = origY;
          origY = image.height - 1 - temp;
        }

        if (isFrontCameraRotated) {
          origX = image.width - 1 - origX;
        }

        origX = origX.clamp(0, image.width - 1);
        origY = origY.clamp(0, image.height - 1);

        final yValue = yPlane[origY * yRowStride + origX] / 255.0;

        if (modelConfig.isColor) {
          final uvX = (origX / 2).floor();
          final uvY = (origY / 2).floor();
          final uvIndex = uvY * uvRowStride + uvX * uvPixelStride;

          final uValue = uvPlane[uvIndex] / 255.0;
          final vValue = uvPlane[uvIndex + 1] / 255.0;

          final r = (yValue + 1.402 * (vValue - 0.5)).clamp(0.0, 1.0);
          final g = (yValue - 0.344136 * (uValue - 0.5) - 0.714136 * (vValue - 0.5)).clamp(0.0, 1.0);
          final b = (yValue + 1.772 * (uValue - 0.5)).clamp(0.0, 1.0);

          inputBuffer[pixelIndex++] = r;
          inputBuffer[pixelIndex++] = g;
          inputBuffer[pixelIndex++] = b;

          debugImage.setPixelRgb(x, y,
              (r * 255).toInt(),
              (g * 255).toInt(),
              (b * 255).toInt());
        } else {
          inputBuffer[pixelIndex++] = yValue;
          debugImage.setPixelRgb(x, y,
              (yValue * 255).toInt(),
              (yValue * 255).toInt(),
              (yValue * 255).toInt());
        }
      }
    }

    if (_saveDebugImages) {
      _saveDebugImage(debugImage);
    }

    debugPrint("✅ Input do modelo preparado");
    return inputBuffer;
  }

  void _processCameraImage(CameraImage image) async {
    final now = DateTime.now();
    if (lastProcessedTime != null && now.difference(lastProcessedTime!).inMilliseconds < 200) {
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
          debugPrint("👤 Rosto detectado no frame em ${message.rect}");
          setState(() {
            faceDetected = true;
            currentFaceRegion = message;
          });

          debugPrint("🧠 Preparando para análise de emoções...");
          final input = _yuvToModelInput(image, message);

          if (_inputShape == null || _outputShape == null) {
            debugPrint("⚠️ Shapes de input/output não disponíveis");
            setState(() => detectedEmotion = "Erro no modelo");
            return;
          }

          final outputSize = _outputShape!.reduce((a, b) => a * b);
          final output = List.filled(outputSize, 0.0).reshape(_outputShape!);

          try {
            debugPrint("⚡ Executando inferência do modelo...");
            _interpreter.run(input.reshape(_inputShape!), output);

            debugPrint("📊 Saídas do modelo: ${output[0]}");

            final modelConfig = modelConfigs[selectedModel]!;
            final labels = modelConfig.labels;
            final labelTranslations = modelConfig.labelTranslations;

            if (output[0].length != labels.length) {
              debugPrint("❌ Número de saídas (${output[0].length}) não corresponde ao número de labels (${labels.length})");
              setState(() => detectedEmotion = "Erro: modelo/labels incompatíveis");
              return;
            }

            int maxIndex = 0;
            double maxConfidence = output[0][0];
            for (int i = 1; i < output[0].length; i++) {
              if (output[0][i] > maxConfidence) {
                maxConfidence = output[0][i];
                maxIndex = i;
              }
            }

            String emotion = labels[maxIndex];
            String emotionTranslated = labelTranslations[emotion] ?? 'Desconhecido';
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
            currentFaceRegion = null;
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
    debugPrint("🖌️ Reconstruindo interface...");
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
        actions: [
          PopupMenuButton<String>(
            onSelected: _changeModel,
            itemBuilder: (BuildContext context) {
              return modelConfigs.keys.map((String model) {
                return PopupMenuItem<String>(
                  value: model,
                  child: Text(model),
                );
              }).toList();
            },
            icon: const Icon(Icons.model_training),
          ),
        ],
      ),
      body: Stack(
        children: [
          if (controller != null && controller!.value.isInitialized)
            CameraPreview(controller!),
          if (currentFaceRegion != null) _buildFaceOverlay(),
        ],
      ),
      floatingActionButton: Column(
        mainAxisAlignment: MainAxisAlignment.end,
        children: [
          FloatingActionButton(
            onPressed: _toggleCamera,
            tooltip: 'Alternar câmera',
            child: const Icon(Icons.switch_camera),
          ),
          SizedBox(height: 16),
          FloatingActionButton(
            onPressed: () {
              setState(() {
                _saveDebugImages = !_saveDebugImages;
                debugPrint(_saveDebugImages
                    ? "🖼️ Salvamento de imagens ativado"
                    : "🖼️ Salvamento de imagens desativado");
              });
            },
            tooltip: 'Debug de imagens',
            child: Icon(_saveDebugImages
                ? Icons.image
                : Icons.image_not_supported),
            heroTag: 'debugImages',
          ),
        ],
      ),
    );
  }

  Widget _buildFaceOverlay() {
    if (currentFaceRegion == null || controller == null) return Container();

    final mediaQuery = MediaQuery.of(context);
    final screenSize = mediaQuery.size;
    final previewSize = controller!.value.previewSize!;
    final isPortrait = _currentOrientation == Orientation.portrait;

    final faceRect = currentFaceRegion!.rect;

    final double scaleX, scaleY;

    if (isPortrait) {
      scaleX = screenSize.width / previewSize.height;
      scaleY = screenSize.height / previewSize.width;
    } else {
      scaleX = screenSize.width / previewSize.width;
      scaleY = screenSize.height / previewSize.height;
    }

    double left = faceRect.left * scaleX;
    double top = faceRect.top * scaleY;
    double right = faceRect.right * scaleX;
    double bottom = faceRect.bottom * scaleY;

    if (isFrontCamera) {
      final tempLeft = left;
      left = screenSize.width - right;
      right = screenSize.width - tempLeft;
    }

    final shrinkFactor = 0.85;
    final centerX = (left + right) / 2;
    final centerY = (top + bottom) / 2;
    final newWidth = (right - left) * shrinkFactor;
    final newHeight = (bottom - top) * shrinkFactor;

    left = centerX - newWidth / 2;
    right = centerX + newWidth / 2;
    top = centerY - newHeight / 2;
    bottom = centerY + newHeight / 2;

    final faceHeight = bottom - top;
    final verticalAdjustment = math.min(faceHeight * 0.15, screenSize.height * 0.05);
    top -= verticalAdjustment;
    bottom -= verticalAdjustment;

    left = left.clamp(0.0, screenSize.width - 10);
    right = right.clamp(10.0, screenSize.width);
    top = top.clamp(0.0, screenSize.height - 10);
    bottom = bottom.clamp(10.0, screenSize.height);

    const margin = 8.0;
    final fontSize = math.max(14.0, (bottom - top) * 0.1);

    return Positioned(
      left: left,
      top: top,
      width: right - left,
      height: bottom - top,
      child: Container(
        decoration: BoxDecoration(
          border: Border.all(
            color: Colors.green.withOpacity(0.8),
            width: 2.5,
          ),
          borderRadius: BorderRadius.circular(6),
        ),
        child: Center(
          child: Padding(
            padding: EdgeInsets.all(margin),
            child: Text(
              detectedEmotion,
              textAlign: TextAlign.center,
              style: TextStyle(
                color: Colors.green,
                fontSize: fontSize,
                fontWeight: FontWeight.bold,
                shadows: [
                  Shadow(
                    color: Colors.black.withOpacity(0.7),
                    blurRadius: 3,
                    offset: Offset(1, 1),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class FacePainter extends CustomPainter {
  final Rect faceRect;
  final List<Offset>? landmarks;

  FacePainter({required this.faceRect, this.landmarks});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.green
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    canvas.drawRect(faceRect, paint);

    if (landmarks != null && landmarks!.isNotEmpty) {
      final landmarkPaint = Paint()
        ..color = Colors.blue
        ..style = PaintingStyle.fill;

      for (final landmark in landmarks!) {
        canvas.drawCircle(landmark, 4.0, landmarkPaint);
      }
    }
  }

  @override
  bool shouldRepaint(FacePainter oldDelegate) {
    return oldDelegate.faceRect != faceRect ||
        !listEquals(oldDelegate.landmarks, landmarks);
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