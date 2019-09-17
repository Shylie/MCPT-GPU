namespace PathTracerGUI
{
    partial class Main
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.pboxPreview = new System.Windows.Forms.PictureBox();
            this.btnRender = new System.Windows.Forms.Button();
            this.listbxMaterials = new System.Windows.Forms.CheckedListBox();
            this.btnAddObj = new System.Windows.Forms.Button();
            this.radioButton1 = new System.Windows.Forms.RadioButton();
            this.radioButton2 = new System.Windows.Forms.RadioButton();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.txtbxFileName = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.txtbxChunkSize = new System.Windows.Forms.TextBox();
            this.txtbxSamples = new System.Windows.Forms.TextBox();
            this.txtbxHeight = new System.Windows.Forms.TextBox();
            this.txtbxWidth = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.label1 = new System.Windows.Forms.Label();
            this.lblRenderTime = new System.Windows.Forms.Label();
            this.pbarDuration = new System.Windows.Forms.ProgressBar();
            this.renderProcess = new System.Diagnostics.Process();
            this.listbxObjects = new System.Windows.Forms.CheckedListBox();
            ((System.ComponentModel.ISupportInitialize)(this.pboxPreview)).BeginInit();
            this.groupBox1.SuspendLayout();
            this.SuspendLayout();
            // 
            // pboxPreview
            // 
            this.pboxPreview.ImageLocation = "C:\\Users\\Buddylgreen\\Source\\Repos\\MCPT-GPU\\PathTracerGUI\\bin\\Release\\Test.ppm";
            this.pboxPreview.Location = new System.Drawing.Point(691, 12);
            this.pboxPreview.Name = "pboxPreview";
            this.pboxPreview.Size = new System.Drawing.Size(759, 629);
            this.pboxPreview.TabIndex = 0;
            this.pboxPreview.TabStop = false;
            // 
            // btnRender
            // 
            this.btnRender.Location = new System.Drawing.Point(170, 589);
            this.btnRender.Name = "btnRender";
            this.btnRender.Size = new System.Drawing.Size(102, 40);
            this.btnRender.TabIndex = 1;
            this.btnRender.Text = "Render";
            this.btnRender.UseVisualStyleBackColor = true;
            this.btnRender.Click += new System.EventHandler(this.BtnRender_Click);
            // 
            // listbxMaterials
            // 
            this.listbxMaterials.FormattingEnabled = true;
            this.listbxMaterials.Location = new System.Drawing.Point(12, 12);
            this.listbxMaterials.Name = "listbxMaterials";
            this.listbxMaterials.Size = new System.Drawing.Size(260, 280);
            this.listbxMaterials.TabIndex = 2;
            // 
            // btnAddObj
            // 
            this.btnAddObj.Enabled = false;
            this.btnAddObj.Location = new System.Drawing.Point(12, 589);
            this.btnAddObj.Name = "btnAddObj";
            this.btnAddObj.Size = new System.Drawing.Size(102, 40);
            this.btnAddObj.TabIndex = 3;
            this.btnAddObj.Text = "Add Object";
            this.btnAddObj.UseVisualStyleBackColor = true;
            this.btnAddObj.Click += new System.EventHandler(this.BtnAddObj_Click);
            // 
            // radioButton1
            // 
            this.radioButton1.AutoSize = true;
            this.radioButton1.Checked = true;
            this.radioButton1.Location = new System.Drawing.Point(6, 75);
            this.radioButton1.Name = "radioButton1";
            this.radioButton1.Size = new System.Drawing.Size(98, 24);
            this.radioButton1.TabIndex = 4;
            this.radioButton1.TabStop = true;
            this.radioButton1.Text = "Chunked";
            this.radioButton1.UseVisualStyleBackColor = true;
            // 
            // radioButton2
            // 
            this.radioButton2.AutoSize = true;
            this.radioButton2.Location = new System.Drawing.Point(6, 105);
            this.radioButton2.Name = "radioButton2";
            this.radioButton2.Size = new System.Drawing.Size(127, 24);
            this.radioButton2.TabIndex = 5;
            this.radioButton2.TabStop = true;
            this.radioButton2.Text = "UnChuncked";
            this.radioButton2.UseVisualStyleBackColor = true;
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.txtbxFileName);
            this.groupBox1.Controls.Add(this.label5);
            this.groupBox1.Controls.Add(this.txtbxChunkSize);
            this.groupBox1.Controls.Add(this.txtbxSamples);
            this.groupBox1.Controls.Add(this.txtbxHeight);
            this.groupBox1.Controls.Add(this.txtbxWidth);
            this.groupBox1.Controls.Add(this.label4);
            this.groupBox1.Controls.Add(this.label3);
            this.groupBox1.Controls.Add(this.label2);
            this.groupBox1.Controls.Add(this.label1);
            this.groupBox1.Controls.Add(this.radioButton1);
            this.groupBox1.Controls.Add(this.radioButton2);
            this.groupBox1.Location = new System.Drawing.Point(278, 12);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(407, 287);
            this.groupBox1.TabIndex = 6;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Render Settings";
            // 
            // txtbxFileName
            // 
            this.txtbxFileName.Location = new System.Drawing.Point(107, 241);
            this.txtbxFileName.Name = "txtbxFileName";
            this.txtbxFileName.Size = new System.Drawing.Size(100, 26);
            this.txtbxFileName.TabIndex = 15;
            this.txtbxFileName.Text = "Test";
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(6, 244);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(80, 20);
            this.label5.TabIndex = 14;
            this.label5.Text = "File Name";
            // 
            // txtbxChunkSize
            // 
            this.txtbxChunkSize.Location = new System.Drawing.Point(301, 134);
            this.txtbxChunkSize.Name = "txtbxChunkSize";
            this.txtbxChunkSize.Size = new System.Drawing.Size(100, 26);
            this.txtbxChunkSize.TabIndex = 13;
            this.txtbxChunkSize.Text = "32";
            // 
            // txtbxSamples
            // 
            this.txtbxSamples.Location = new System.Drawing.Point(301, 102);
            this.txtbxSamples.Name = "txtbxSamples";
            this.txtbxSamples.Size = new System.Drawing.Size(100, 26);
            this.txtbxSamples.TabIndex = 12;
            this.txtbxSamples.Text = "50";
            // 
            // txtbxHeight
            // 
            this.txtbxHeight.Location = new System.Drawing.Point(301, 70);
            this.txtbxHeight.Name = "txtbxHeight";
            this.txtbxHeight.Size = new System.Drawing.Size(100, 26);
            this.txtbxHeight.TabIndex = 11;
            this.txtbxHeight.Text = "512";
            // 
            // txtbxWidth
            // 
            this.txtbxWidth.Location = new System.Drawing.Point(301, 38);
            this.txtbxWidth.Name = "txtbxWidth";
            this.txtbxWidth.Size = new System.Drawing.Size(100, 26);
            this.txtbxWidth.TabIndex = 10;
            this.txtbxWidth.Text = "512";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(205, 137);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(90, 20);
            this.label4.TabIndex = 9;
            this.label4.Text = "Chunk Size";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(224, 105);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(71, 20);
            this.label3.TabIndex = 8;
            this.label3.Text = "Samples";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(239, 73);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(56, 20);
            this.label2.TabIndex = 7;
            this.label2.Text = "Height";
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(245, 41);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(50, 20);
            this.label1.TabIndex = 6;
            this.label1.Text = "Width";
            // 
            // lblRenderTime
            // 
            this.lblRenderTime.AutoSize = true;
            this.lblRenderTime.Location = new System.Drawing.Point(166, 632);
            this.lblRenderTime.Name = "lblRenderTime";
            this.lblRenderTime.Size = new System.Drawing.Size(140, 20);
            this.lblRenderTime.TabIndex = 8;
            this.lblRenderTime.Text = "Last Render Took:";
            // 
            // pbarDuration
            // 
            this.pbarDuration.Location = new System.Drawing.Point(278, 589);
            this.pbarDuration.MarqueeAnimationSpeed = 200;
            this.pbarDuration.Name = "pbarDuration";
            this.pbarDuration.Size = new System.Drawing.Size(291, 40);
            this.pbarDuration.Style = System.Windows.Forms.ProgressBarStyle.Marquee;
            this.pbarDuration.TabIndex = 7;
            this.pbarDuration.UseWaitCursor = true;
            this.pbarDuration.Visible = false;
            // 
            // renderProcess
            // 
            this.renderProcess.StartInfo.Domain = "";
            this.renderProcess.StartInfo.LoadUserProfile = false;
            this.renderProcess.StartInfo.Password = null;
            this.renderProcess.StartInfo.StandardErrorEncoding = null;
            this.renderProcess.StartInfo.StandardOutputEncoding = null;
            this.renderProcess.StartInfo.UserName = "";
            this.renderProcess.SynchronizingObject = this;
            // 
            // listbxObjects
            // 
            this.listbxObjects.FormattingEnabled = true;
            this.listbxObjects.Location = new System.Drawing.Point(12, 316);
            this.listbxObjects.Name = "listbxObjects";
            this.listbxObjects.Size = new System.Drawing.Size(260, 234);
            this.listbxObjects.TabIndex = 9;
            // 
            // Main
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 20F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1462, 679);
            this.Controls.Add(this.listbxObjects);
            this.Controls.Add(this.lblRenderTime);
            this.Controls.Add(this.pbarDuration);
            this.Controls.Add(this.groupBox1);
            this.Controls.Add(this.btnAddObj);
            this.Controls.Add(this.listbxMaterials);
            this.Controls.Add(this.btnRender);
            this.Controls.Add(this.pboxPreview);
            this.Name = "Main";
            this.Text = "Path Tracer";
            ((System.ComponentModel.ISupportInitialize)(this.pboxPreview)).EndInit();
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.PictureBox pboxPreview;
        private System.Windows.Forms.Button btnRender;
        private System.Windows.Forms.CheckedListBox listbxMaterials;
        private System.Windows.Forms.RadioButton radioButton1;
        private System.Windows.Forms.RadioButton radioButton2;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.TextBox txtbxChunkSize;
        private System.Windows.Forms.TextBox txtbxSamples;
        private System.Windows.Forms.TextBox txtbxHeight;
        private System.Windows.Forms.TextBox txtbxWidth;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox txtbxFileName;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.Label lblRenderTime;
        private System.Windows.Forms.ProgressBar pbarDuration;
        private System.Diagnostics.Process renderProcess;
        private System.Windows.Forms.CheckedListBox listbxObjects;
        private System.Windows.Forms.Button btnAddObj;
    }
}

