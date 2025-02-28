import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

late List<CameraDescription> _cameras;

void main() async{
  WidgetsFlutterBinding.ensureInitialized();
  _cameras = await availableCameras();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'Flutter Demo Home Page'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;
  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  late CameraController controller;
  late FaceDetector faceDetector;
  DateTime? lastProcessedTime; // Armazena o tempo do último frame processado
  bool faceDetected = false; // Variável para indicar se um rosto foi encontrado
  List<Face> faces = []; // Lista para armazenar rostos detectados
  bool isProcessing = false;


  @override
  void initState() {
    super.initState();
    faceDetector = FaceDetector(
      options: FaceDetectorOptions(
        enableContours: false,
        enableLandmarks: false,
        performanceMode: FaceDetectorMode.accurate,
        enableTracking: true,
      ),
    );
    controller = CameraController(_cameras[1], ResolutionPreset.high);
    controller.initialize().then((_) {
      if (!mounted) {
        return;
      }

      controller.startImageStream((CameraImage image) async {
        DateTime now = DateTime.now();

        // Processa apenas 1 frame por segundo
        if (lastProcessedTime == null || now.difference(lastProcessedTime!).inMilliseconds >= 500) {
          lastProcessedTime = now;
        }

        if (isProcessing) return; // Ignora novos frames se já estiver processando
        isProcessing = true;
        await detectFaces(image);
        isProcessing = false;
      });

      setState(() {});
    }).catchError((Object e) {
      if (e is CameraException) {
        switch (e.code) {
          case 'CameraAccessDenied':
          // Handle access errors here.
            break;
          default:
          // Handle other errors here.
            break;
        }
      }
    });
  }

  Future<void> detectFaces(CameraImage image) async {
    final inputImage = InputImage.fromBytes(
      bytes: concatenatePlanes(image.planes),
      metadata: InputImageMetadata(
        size: Size(image.width.toDouble(), image.height.toDouble()),
        rotation: InputImageRotation.rotation0deg,
        format: InputImageFormat.nv21,
        bytesPerRow: image.planes[0].bytesPerRow,
      ),
    );

    final List<Face> faces = await faceDetector.processImage(inputImage);
    setState(() {
      faceDetected = faces.isNotEmpty; // Atualiza o estado com base na detecção
    });

    if (faceDetected) {
      debugPrint("Rosto detectado!");
    } else {
      debugPrint("Nenhum rosto encontrado.");
    }
  }

  Uint8List concatenatePlanes(List<Plane> planes) {
    final WriteBuffer allBytes = WriteBuffer();
    for (Plane plane in planes) {
      allBytes.putUint8List(plane.bytes);
    }
    return allBytes.done().buffer.asUint8List();
  }

  @override
  void dispose() {
    controller.dispose();
    faceDetector.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(widget.title),
      ),
      body: Stack(
        children: [
          // Exibe a câmera
          Positioned.fill(
            child: controller.value.isInitialized
                ? CameraPreview(controller)
                : Center(child: CircularProgressIndicator()),
          ),

          // Texto indicando se o rosto foi detectado
          Positioned(
            bottom: 50,
            left: 0,
            right: 0,
            child: Center(
              child: Container(
                padding: EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: faceDetected ? Colors.green.withOpacity(0.7) : Colors.red.withOpacity(0.7),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Text(
                  faceDetected ? "Rosto Detectado!" : "Nenhum rosto encontrado",
                  style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
