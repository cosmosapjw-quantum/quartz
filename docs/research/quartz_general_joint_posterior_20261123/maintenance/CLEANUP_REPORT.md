# C27 authorized workspace and backup consolidation

The user requested removal of unused worktrees and superseded backups while retaining necessary compressed material in the latest backup. This instruction supersedes the initial all-checkouts-preserved delivery assumption. The scientific SSOT remains87c15dc828d6eea519c219fd8a412fa21dfc464d.

## Actual removal

-12 clean, unlocked, inactive linked worktrees were removed with git worktree remove, without force. Each HEAD matched a live remote branch and the independently restored all-ref bundle. All12 branches/commits remain available.
-11 obsolete backup directories were removed after SHA256/size/metadata verification of432 individual files. Only empty directories were subsequently removed.
-19 superseded archive copies in the current C27 backup were removed after verified full-source and history consolidation. These include15 C27 stage overlays and4 copied predecessor archives.
-C24 had3 ignored scientific trace gzips, total262744237 bytes, absent from reachable Git. Their original compressed bytes were retained in the unique-payload archive and checked by SHA256/readback; original gzip CRC/EOF was checked as well.

## Retained recovery set

The latest backup folder is /home/cosmosapjw/Dropbox/personal_projects/quartz_general_meta27_backups. It contains a complete C05 source snapshot, a Git bundle carrying all102 captured refs, retained_history with30 unique non-Git payloads and original-path/member mapping, and stage/publication/cleanup receipts. The full snapshot was checked against2863 Git blobs and file modes. The bundle passed git bundle verify, fresh mirror clone, complete fsck, ref-by-ref identity and source-tree comparison. Each of425 old archive inputs was read and mapped to Git or retained content before deletion.

Original tar/gzip padding/header byte identity is not retained. Research payload bytes, member metadata, history and provenance mappings are retained. Git-backed files are restored by commit/blob identity; non-Git files are restored from content-addressed members. Later C06-C10 stages use the consolidated full base plus current overlays; final C10 consolidation will replace the intermediate complete snapshot/bundle.

## Space accounting

Removed worktree inventory allocated bytes: 11,861,192,704.
Original12-backup inventory allocated bytes: 11,318,706,176.
Current consolidated backup allocated bytes at this checkpoint: 2,183,946,240.
Path-occupancy reduction, after charging the new recovery set: 20,995,952,640 bytes.
Filesystem free bytes at this checkpoint: 218,729,267,200.

These quantities are not a causal estimate of the total filesystem free-space change. Concurrent unrelated activity changed disk space during the task; no external increase is counted as our cleanup.

## Protected and unresolved scope

The original repository and active C26/C27 worktrees were not deletion targets. Original source/status and C26 identity match the baseline. Original non-Git research data remains outside this cleanup. The preexisting dirty reboot checkout was found absent without any deletion command or receipt from this run. Its preservation is external_or_unresolved; this report does not claim its uncommitted contents were recovered.

The all-ref bundle preserves the original history, so removing redundant checkouts does not remove the historical motivation or the original SSOT. Scientific gain, implementation correctness and delivery preservation remain separate outcomes. Provider-side Dropbox sync acknowledgment has not been verified.
