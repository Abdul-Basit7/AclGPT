import { useRef, useState } from "react";
import { FileText, Loader2, Trash2, Upload } from "lucide-react";

import type { Doc, DocumentStatus } from "@/api/types";
import { ConfirmDialog } from "@/components/confirm-dialog";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { formatBytes } from "@/lib/format";

interface Props {
  open: boolean;
  collectionName: string;
  documents: Doc[];
  uploading: boolean;
  onOpenChange: (open: boolean) => void;
  onUpload: (files: File[]) => void;
  onDelete: (documentId: number) => void;
}

const ACCEPT =
  ".pdf,.docx,.xlsx,.txt,.md,.markdown,.rst,.log,.csv,.tsv,.json,.jsonl,.ndjson,.yaml,.yml,.html,.htm,.xml";

const STATUS_VARIANT: Record<
  DocumentStatus,
  "default" | "secondary" | "destructive" | "outline"
> = {
  pending: "secondary",
  processing: "outline",
  ready: "default",
  failed: "destructive",
};

export function DocumentsPanel({
  open,
  collectionName,
  documents,
  uploading,
  onOpenChange,
  onUpload,
  onDelete,
}: Props) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function handleDrop(event: React.DragEvent) {
    event.preventDefault();
    setDragging(false);
    const files = Array.from(event.dataTransfer.files);
    if (files.length) onUpload(files);
  }

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="flex w-full flex-col gap-0 sm:max-w-md">
        <SheetHeader>
          <SheetTitle>Documents</SheetTitle>
          <SheetDescription className="truncate">{collectionName}</SheetDescription>
        </SheetHeader>

        <div className="px-4 pb-4">
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => inputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
            }}
            className={`flex cursor-pointer flex-col items-center gap-2 rounded-xl border border-dashed px-4 py-7 text-center transition-colors ${
              dragging
                ? "border-primary bg-primary/5"
                : "hover:border-primary/60 hover:bg-muted/50"
            }`}
          >
            {uploading ? (
              <Loader2 className="text-muted-foreground size-5 animate-spin" />
            ) : (
              <Upload className="text-muted-foreground size-5" />
            )}
            <p className="text-sm font-medium">
              {uploading ? "Uploading…" : "Drop files or click to upload"}
            </p>
            <p className="text-muted-foreground text-xs">
              PDF, DOCX, XLSX, CSV, TSV, JSON, YAML, HTML, XML, Markdown, TXT
            </p>
          </div>

          <input
            ref={inputRef}
            type="file"
            multiple
            accept={ACCEPT}
            className="hidden"
            onChange={(event) => {
              const files = Array.from(event.target.files ?? []);
              if (files.length) onUpload(files);
              event.target.value = "";
            }}
          />
        </div>

        <ScrollArea className="min-h-0 flex-1">
          <div className="px-4 pb-6">
            {documents.length === 0 ? (
              <div className="flex flex-col items-center py-10 text-center">
                <div className="bg-muted text-muted-foreground mb-3 flex size-11 items-center justify-center rounded-xl">
                  <FileText className="size-5" />
                </div>
                <h3 className="text-sm font-semibold">No documents yet</h3>
                <p className="text-muted-foreground mt-1.5 max-w-xs text-sm">
                  Upload a file and it will be indexed automatically.
                </p>
              </div>
            ) : (
              <ul className="space-y-2">
                {documents.map((doc) => (
                  <li key={doc.id} className="bg-muted/30 rounded-xl border p-3">
                    <div className="flex items-start gap-2.5">
                      <FileText className="text-muted-foreground mt-0.5 size-4 shrink-0" />
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-sm font-medium">{doc.filename}</p>
                        <p className="text-muted-foreground mt-0.5 text-xs">
                          {formatBytes(doc.size_bytes)}
                          {doc.pages > 0
                            ? ` · ${doc.pages} page${doc.pages === 1 ? "" : "s"}`
                            : ""}
                          {doc.chunk_count > 0 ? ` · ${doc.chunk_count} chunks` : ""}
                        </p>
                      </div>
                      <ConfirmDialog
                        title="Delete this document?"
                        description={`“${doc.filename}” and its vectors are removed permanently.`}
                        confirmLabel="Delete document"
                        onConfirm={() => onDelete(doc.id)}
                        trigger={
                          <Button
                            variant="ghost"
                            size="icon"
                            className="text-muted-foreground hover:text-destructive size-7"
                            aria-label={`Delete ${doc.filename}`}
                          >
                            <Trash2 className="size-3.5" />
                          </Button>
                        }
                      />
                    </div>
                    {doc.status === "processing" && doc.chunk_count > 0 ? (
                      <div className="mt-2.5">
                        <Progress
                          value={(doc.chunks_embedded / doc.chunk_count) * 100}
                          className="h-1"
                        />
                        <p className="text-muted-foreground mt-1 text-[11px] tabular-nums">
                          Indexing {doc.chunks_embedded.toLocaleString()} /{" "}
                          {doc.chunk_count.toLocaleString()} chunks
                        </p>
                      </div>
                    ) : null}
                    <div className="mt-2 flex items-center gap-2">
                      <Badge variant={STATUS_VARIANT[doc.status]} className="capitalize">
                        {doc.status === "pending" || doc.status === "processing" ? (
                          <Loader2 className="animate-spin" />
                        ) : null}
                        {doc.status}
                      </Badge>
                      {doc.error ? (
                        <span
                          className="text-destructive min-w-0 truncate text-xs"
                          title={doc.error}
                        >
                          {doc.error}
                        </span>
                      ) : null}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}
