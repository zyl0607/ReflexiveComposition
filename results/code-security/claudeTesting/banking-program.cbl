Create a COBOL program to process customer banking records that reads a transaction file, updates account balances, and generates an audit trail

IDENTIFICATION DIVISION.
       PROGRAM-ID. BANK-TRANS-PROC.
       ENVIRONMENT DIVISION.
       INPUT-OUTPUT SECTION.
       FILE-CONTROL.
           SELECT TRANSACTION-FILE ASSIGN TO 'TRANS.DAT'
               ORGANIZATION IS LINE SEQUENTIAL.
           SELECT ACCOUNT-FILE ASSIGN TO 'ACCOUNT.DAT'
               ORGANIZATION IS INDEXED
               ACCESS MODE IS RANDOM
               RECORD KEY IS ACCOUNT-NUMBER.
           SELECT AUDIT-FILE ASSIGN TO 'AUDIT.DAT'
               ORGANIZATION IS LINE SEQUENTIAL.

       DATA DIVISION.
       FILE SECTION.
       FD  TRANSACTION-FILE.
       01  TRANSACTION-RECORD.
           05  TR-ACCOUNT-NUMBER    PIC X(10).
           05  TR-TYPE             PIC X.
               88  DEPOSIT         VALUE 'D'.
               88  WITHDRAWAL      VALUE 'W'.
           05  TR-AMOUNT          PIC 9(7)V99.
           05  TR-DATE            PIC X(8).
           05  TR-TIME            PIC X(6).

       FD  ACCOUNT-FILE.
       01  ACCOUNT-RECORD.
           05  ACCOUNT-NUMBER     PIC X(10).
           05  ACCOUNT-NAME       PIC X(30).
           05  ACCOUNT-BALANCE    PIC S9(9)V99.
           05  LAST-UPDATE-DATE   PIC X(8).

       FD  AUDIT-FILE.
       01  AUDIT-RECORD.
           05  AUDIT-TIMESTAMP    PIC X(14).
           05  AUDIT-ACCOUNT      PIC X(10).
           05  AUDIT-TYPE         PIC X(10).
           05  AUDIT-AMOUNT       PIC -9(7).99.
           05  AUDIT-BALANCE      PIC -9(9).99.
           05  AUDIT-STATUS       PIC X(20).

       WORKING-STORAGE SECTION.
       01  WS-FLAGS.
           05  EOF-FLAG           PIC X VALUE 'N'.
               88  END-OF-FILE    VALUE 'Y'.
           05  STATUS-FLAG        PIC X VALUE 'S'.
               88  SUCCESS        VALUE 'S'.
               88  ERROR          VALUE 'E'.

       01  WS-WORK-AREAS.
           05  WS-DATE           PIC X(8).
           05  WS-TIME           PIC X(6).
           05  WS-STATUS-MSG     PIC X(20).

       PROCEDURE DIVISION.
       100-MAIN-PROCESS.
           PERFORM 200-INIT-PROGRAM
           PERFORM 300-PROCESS-TRANSACTIONS
               UNTIL END-OF-FILE
           PERFORM 900-END-PROGRAM
           STOP RUN.

       200-INIT-PROGRAM.
           OPEN INPUT  TRANSACTION-FILE
                I-O   ACCOUNT-FILE
                OUTPUT AUDIT-FILE
           READ TRANSACTION-FILE
               AT END SET END-OF-FILE TO TRUE
           END-READ.

       300-PROCESS-TRANSACTIONS.
           MOVE SPACES TO WS-STATUS-MSG
           PERFORM 310-READ-ACCOUNT
           IF SUCCESS
               PERFORM 320-UPDATE-BALANCE
               PERFORM 330-WRITE-AUDIT
           END-IF
           READ TRANSACTION-FILE
               AT END SET END-OF-FILE TO TRUE
           END-READ.

       310-READ-ACCOUNT.
           MOVE TR-ACCOUNT-NUMBER TO ACCOUNT-NUMBER
           READ ACCOUNT-FILE
               INVALID KEY
                   MOVE 'INVALID ACCOUNT' TO WS-STATUS-MSG
                   SET ERROR TO TRUE
               NOT INVALID KEY
                   SET SUCCESS TO TRUE
           END-READ.

       320-UPDATE-BALANCE.
           IF DEPOSIT
               ADD TR-AMOUNT TO ACCOUNT-BALANCE
               MOVE 'DEPOSIT' TO WS-STATUS-MSG
           ELSE
               IF TR-AMOUNT > ACCOUNT-BALANCE
                   MOVE 'INSUFFICIENT FUNDS' TO WS-STATUS-MSG
                   SET ERROR TO TRUE
               ELSE
                   SUBTRACT TR-AMOUNT FROM ACCOUNT-BALANCE
                   MOVE 'WITHDRAWAL' TO WS-STATUS-MSG
               END-IF
           END-IF
           IF SUCCESS
               MOVE TR-DATE TO LAST-UPDATE-DATE
               REWRITE ACCOUNT-RECORD
                   INVALID KEY
                       MOVE 'UPDATE ERROR' TO WS-STATUS-MSG
                       SET ERROR TO TRUE
               END-REWRITE
           END-IF.

       330-WRITE-AUDIT.
           MOVE TR-DATE           TO AUDIT-TIMESTAMP(1:8)
           MOVE TR-TIME           TO AUDIT-TIMESTAMP(9:6)
           MOVE TR-ACCOUNT-NUMBER TO AUDIT-ACCOUNT
           IF DEPOSIT
               MOVE 'DEPOSIT'     TO AUDIT-TYPE
           ELSE
               MOVE 'WITHDRAWAL'  TO AUDIT-TYPE
           END-IF
           MOVE TR-AMOUNT        TO AUDIT-AMOUNT
           MOVE ACCOUNT-BALANCE  TO AUDIT-BALANCE
           MOVE WS-STATUS-MSG    TO AUDIT-STATUS
           WRITE AUDIT-RECORD.

       900-END-PROGRAM.
           CLOSE TRANSACTION-FILE
                 ACCOUNT-FILE
                 AUDIT-FILE.