package com.o19s.es.ltr.ranker.tensorflow;

import org.tensorflow.SavedModelBundle;

import java.io.ByteArrayInputStream;
import java.io.Closeable;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.FileVisitResult;
import java.nio.file.FileVisitor;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.Base64;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

public class TensorflowModelLoader implements Closeable {
    private static final int READ_BUFFER_SIZE = 128 * 1024;
    private final Path modelDir;

    public TensorflowModelLoader(Path baseModelDir) throws IOException {
        modelDir = Files.createTempDirectory(baseModelDir, getClass().getSimpleName());
    }

    public SavedModelBundle load(String base64Model, String tag) throws IOException {
        extractToDirectory(modelDir, base64Model);
        return SavedModelBundle.load(modelDir.toString(), tag);
    }

    public void close() throws IOException {
        deleteDirectoryRecursive(modelDir);
    }

    private void extractToDirectory(Path dir, String modelBase64) throws IOException {
        // Tensorflow wants to read from a directory. Make it happy, take
        // in the model as a base64 encoded zip file and extract it to
        // a temp path.
        byte[] modelBytes = Base64.getDecoder().decode(modelBase64);
        ByteArrayInputStream bais = new ByteArrayInputStream(modelBytes);
        try(ZipInputStream zis = new ZipInputStream(bais)) {
            while (true) {
                ZipEntry entry = zis.getNextEntry();
                if (entry == null) {
                    break;
                }
                Path dest = dir.resolve(entry.getName());
                if (entry.isDirectory()) {
                    Files.createDirectories(dest);
                } else {
                    writeFile(dest, zis);
                }
            }
        }
    }

    private void writeFile(Path path, ZipInputStream is) throws IOException {
        try (OutputStream os = Files.newOutputStream(path)) {
            byte[] buffer = new byte[READ_BUFFER_SIZE];
            int length;
            while ((length = is.read(buffer)) > 0) {
                os.write(buffer, 0, length);
            }
        }
    }

    private void deleteDirectoryRecursive(Path dir) throws IOException {
        Files.walkFileTree(dir, new FileVisitor<Path>() {
            @Override
            public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs) {
                return FileVisitResult.CONTINUE;
            }

            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                Files.delete(file);
                return FileVisitResult.CONTINUE;
            }

            @Override
            public FileVisitResult visitFileFailed(Path file, IOException exc) {
                return FileVisitResult.CONTINUE;
            }

            @Override
            public FileVisitResult postVisitDirectory(Path dir, IOException exc) throws IOException {
                Files.delete(dir);
                return FileVisitResult.CONTINUE;
            }
        });
    }
}
