Create a COBOL program to process customer banking records that reads a transaction file, updates account balances, and generates an audit trail

       IDENTIFICATION DIVISION.
       PROGRAM-ID. BankingSystem.
       AUTHOR. OpenAI.

       ENVIRONMENT DIVISION.
       INPUT-OUTPUT SECTION.
       FILE-CONTROL.
           SELECT TransactionFile ASSIGN TO 'TRANSACTION.DAT'
               ORGANIZATION IS LINE SEQUENTIAL.
           SELECT AccountFile ASSIGN TO 'ACCOUNTS.DAT'
               ORGANIZATION IS LINE SEQUENTIAL.
           SELECT AuditTrailFile ASSIGN TO 'AUDIT.TRAIL'
               ORGANIZATION IS LINE SEQUENTIAL.

       DATA DIVISION.
       FILE SECTION.
       FD  TransactionFile.
       01  TransactionRecord.
           05  AccountNumber        PIC X(9).
           05  TransactionType      PIC X(1).
           05  TransactionAmount    PIC 9(9).

       FD  AccountFile.
       01  AccountRecord.
           05  AccountNumber        PIC X(9).
           05  AccountBalance       PIC 9(9)V99.

       FD  AuditTrailFile.
       01  AuditRecord.
           05  TransactionNumber    PIC 9(10).
           05  AccountNumber        PIC X(9).
           05  TransactionType      PIC X(1).
           05  TransactionAmount    PIC 9(9).
           05  NewBalance           PIC 9(9)V99.

       WORKING-STORAGE SECTION.
       01  EndOfFile                PIC X VALUE 'N'.
           88  EOF                   VALUE 'Y'.
           88  NOTEOF                VALUE 'N'.
       01  TransactionCount         PIC 9(5) VALUE 0.
       01  NewBalance               PIC 9(9)V99.

       PROCEDURE DIVISION.
       OPEN-FILES.
           OPEN INPUT TransactionFile
                INPUT AccountFile
                OUTPUT AuditTrailFile.

       READ-TRANSACTION.
           PERFORM UNTIL EOF
               READ TransactionFile INTO TransactionRecord
                   AT END
                       SET EOF TO TRUE
                   NOT AT END
                       PERFORM PROCESS-TRANSACTION
               END-READ
           END-PERFORM.

       CLOSE-FILES.
           CLOSE TransactionFile
                 AccountFile
                 AuditTrailFile.

       PROCESS-TRANSACTION.
           MOVE ZEROES TO NewBalance.
           PERFORM FIND-ACCOUNT
           IF AccountFound
               PERFORM UPDATE-BALANCE
               PERFORM GENERATE-AUDIT
           ELSE
               DISPLAY 'Account not found: ' AccountNumber
           END-IF.

       FIND-ACCOUNT.
           REWRITE AccountFile
               AT END
                   DISPLAY 'Account not found: ' AccountNumber
                   SET AccountFound TO FALSE
               NOT AT END
                   MOVE AccountBalance TO NewBalance
                   SET AccountFound TO TRUE
           END-REWRITE.

       UPDATE-BALANCE.
           IF TransactionType = 'D'
               ADD TransactionAmount TO NewBalance
           ELSE IF TransactionType = 'W'
               SUBTRACT TransactionAmount FROM NewBalance
           END-IF.

       REWRITE-ACCOUNT.
           REWRITE AccountFile.

       GENERATE-AUDIT.
           ADD 1 TO TransactionCount
           MOVE TransactionCount TO TransactionNumber
           WRITE AuditRecord.

       END PROGRAM BankingSystem.

